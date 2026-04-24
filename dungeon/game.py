"""Game loop: turn scheduling, message delivery, end conditions.

Turn model: one integer `ws.turn` per full round. Both agents act on the
same turn number in strict A-then-B order. Messages sent on turn N are
delivered at the START of turn N+1, before either agent acts. This matches
the spec's "delivered on the following turn, not instantly" rule and means
the delay is experienced even if B acts directly after A on the same turn N.

Phase 2 note: the loop takes pre-action snapshots of belief and world
truth, then feeds them plus the tool execution record into the event
logger. This is the only point where decision-time state is captured;
doing it anywhere else would let it drift with the mutations the tool
itself performs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

from .agent import DungeonAgent
from .config import DEFAULT_WORLD_CONFIG, WorldConfig
from .events import EventLogger
from .tools import DIRECTION_DELTAS
from .tracing import MultiSink
from .world import Message, Pos, WorldState, generate_world


class StepLogger(Protocol):
    def log_step(self, ws: WorldState, agent: DungeonAgent, record: dict) -> None: ...
    def log_end(self, ws: WorldState) -> None: ...


@dataclass
class ConsoleLogger:
    """Minimal stdout printer for human-readable local debugging."""

    quiet: bool = False

    def log_step(self, ws: WorldState, agent: DungeonAgent, record: dict) -> None:
        if self.quiet:
            return
        tool = record.get("tool_name") or "(no tool)"
        args = record.get("tool_input") or {}
        success = "ok" if record.get("semantic_success") else "noop"
        pos = ws.agent_positions[agent.agent_id]
        reason = ""
        r = record.get("tool_result", {})
        if not record.get("semantic_success"):
            reason = r.get("reason") or r.get("note") or ""
        reasoning = (record.get("reasoning") or "").replace("\n", " ")
        if len(reasoning) > 80:
            reasoning = reasoning[:77] + "..."
        print(
            f"turn {ws.turn:02d} {agent.agent_id} @{pos} "
            f"{tool}({_fmt_args(args)}) -> {success} {reason}".rstrip()
        )
        if reasoning:
            print(f"         reasoning: {reasoning}")

    def log_end(self, ws: WorldState) -> None:
        if self.quiet:
            return
        print(f"\n=== game ended: {ws.status} after turn {ws.turn} ===")


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _move_target(pos: Pos, tool_input: dict) -> Pos | None:
    direction = tool_input.get("direction")
    delta = DIRECTION_DELTAS.get(direction) if direction else None
    if delta is None:
        return None
    return (pos[0] + delta[0], pos[1] + delta[1])


def _serialize_message(m: Message) -> dict:
    return {
        "sender": m.sender,
        "recipient": m.recipient,
        "content": m.content,
        "sent_turn": m.sent_turn,
        "deliver_turn": m.deliver_turn,
    }


def deliver_messages(ws: WorldState) -> None:
    """Move any pending messages whose deliver_turn has arrived into inboxes."""
    still_pending = []
    for m in ws.pending_messages:
        if m.deliver_turn <= ws.turn:
            ws.inboxes[m.recipient].append(m)
        else:
            still_pending.append(m)
    ws.pending_messages = still_pending


def _both_at_exit(ws: WorldState) -> bool:
    return all(p == ws.exit_position for p in ws.agent_positions.values())


def _check_end(ws: WorldState, turn_limit: int) -> bool:
    if _both_at_exit(ws):
        ws.status = "success"
        return True
    if ws.turn >= turn_limit:
        ws.status = "timeout"
        return True
    if all(c >= ws.config.stuck_threshold for c in ws.stuck_counter.values()):
        ws.status = "stuck"
        return True
    return False


def run_game(
    seed: int,
    client: Any,
    model: str,
    run_id: str,
    turn_limit: int | None = None,
    console_logger: StepLogger | None = None,
    event_logger: EventLogger | None = None,
    tracer: MultiSink | None = None,
    agent_ids: list[str] | None = None,
    config: WorldConfig = DEFAULT_WORLD_CONFIG,
) -> WorldState:
    agent_ids = agent_ids or ["A", "B"]
    # An explicit turn_limit kwarg overrides the value baked into config,
    # so existing CLI invocations (which pass turn_limit directly) keep
    # working unchanged.
    if turn_limit is None:
        turn_limit = config.turn_limit
    ws = generate_world(seed, agent_ids, config)
    agents = {aid: DungeonAgent(aid, client, model, ws.agent_positions[aid]) for aid in agent_ids}
    if tracer is not None:
        tracer.start_run(
            run_id=run_id,
            seed=seed,
            model=model,
            metadata={
                "turn_limit": turn_limit,
                "agent_ids": agent_ids,
                "initial_positions": {aid: list(p) for aid, p in ws.agent_positions.items()},
            },
        )

    while True:
        if _check_end(ws, turn_limit):
            break

        deliver_messages(ws)

        for aid in agent_ids:
            agent = agents[aid]

            # Snapshot decision-time state BEFORE the tool mutates anything.
            belief_before = agent.belief.snapshot()
            truth_before = ws.ground_truth_snapshot()
            provenance_before = copy.deepcopy(ws.provenance)
            received_preview = [_serialize_message(m) for m in ws.inboxes[aid]]
            pending_before = len(ws.pending_messages)

            record = agent.take_turn(ws)

            sent_this_turn = [_serialize_message(m) for m in ws.pending_messages[pending_before:]]
            unread_after = len(ws.inboxes[aid])

            if record.get("tool_name") == "move":
                if record.get("semantic_success"):
                    ws.stuck_counter[aid] = 0
                else:
                    # Count failed moves where the agent should have known
                    # better: the target is either a cell it has already seen
                    # (so it observed the wall) or it is out of bounds (and
                    # the grid edge is a hard knowable boundary the first
                    # time the agent stands on it). Failed moves into unseen
                    # in-bounds cells are legitimate exploration and do not
                    # count, matching the environment_constraint category
                    # the Phase 2 classifier uses.
                    target = _move_target(ws.agent_positions[aid], record.get("tool_input") or {})
                    if target is not None and (
                        target in agent.belief.seen_cells or not ws.in_bounds(target)
                    ):
                        ws.stuck_counter[aid] += 1

            ws.agents_at_exit = {a for a, p in ws.agent_positions.items() if p == ws.exit_position}

            event = None
            if event_logger is not None:
                event = event_logger.log_step(
                    agent_id=aid,
                    turn=ws.turn,
                    belief_snapshot=belief_before,
                    truth_snapshot=truth_before,
                    provenance=provenance_before,
                    record=record,
                    received_messages=received_preview,
                    sent_messages=sent_this_turn,
                    unread_inbox_count=unread_after,
                )

            if tracer is not None:
                tracer.start_turn(turn=ws.turn, agent_id=aid)
                tracer.log_llm_call(
                    system=record.get("system_prompt") or "",
                    user_prompt=record.get("user_prompt") or "",
                    output_blocks=record.get("response_content_blocks") or [],
                    usage=record.get("usage"),
                    latency_ms=record.get("llm_latency_ms"),
                    model=record.get("model") or model,
                )
                if record.get("tool_name"):
                    tracer.log_tool_call(
                        name=record["tool_name"],
                        tool_input=record.get("tool_input") or {},
                        output=record.get("tool_result") or {},
                        latency_ms=record.get("tool_latency_ms"),
                    )
                tracer.end_turn(
                    outcome={
                        "semantic_success": record.get("semantic_success"),
                        "failure_classification": event.failure_classification if event else None,
                        "divergence_fields": event.divergence_fields if event else [],
                    }
                )

            if console_logger is not None:
                console_logger.log_step(ws, agent, record)

            if _check_end(ws, turn_limit):
                break

        if ws.status != "running":
            break
        ws.turn += 1

    if console_logger is not None:
        console_logger.log_end(ws)
    if tracer is not None:
        tracer.end_run(
            status=ws.status,
            summary={
                "turns_played": ws.turn,
                "final_positions": {aid: list(p) for aid, p in ws.agent_positions.items()},
                "final_inventories": {
                    aid: sorted(inv) for aid, inv in ws.agent_inventories.items()
                },
                "door_locked": ws.door_locked,
                "key_holder": ws.key_holder,
            },
        )
    return ws
