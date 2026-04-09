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
from typing import Any, Optional, Protocol

from .agent import DungeonAgent
from .events import EventLogger
from .world import DEFAULT_TURN_LIMIT, STUCK_THRESHOLD, Message, WorldState, generate_world


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
    if all(c >= STUCK_THRESHOLD for c in ws.stuck_counter.values()):
        ws.status = "stuck"
        return True
    return False


def run_game(
    seed: int,
    client: Any,
    model: str,
    run_id: str,
    turn_limit: int = DEFAULT_TURN_LIMIT,
    console_logger: Optional[StepLogger] = None,
    event_logger: Optional[EventLogger] = None,
    agent_ids: Optional[list[str]] = None,
) -> WorldState:
    agent_ids = agent_ids or ["A", "B"]
    ws = generate_world(seed, agent_ids)
    agents = {
        aid: DungeonAgent(aid, client, model, ws.agent_positions[aid])
        for aid in agent_ids
    }

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

            sent_this_turn = [
                _serialize_message(m) for m in ws.pending_messages[pending_before:]
            ]
            unread_after = len(ws.inboxes[aid])

            if record.get("tool_name") == "move":
                if record.get("semantic_success"):
                    ws.stuck_counter[aid] = 0
                else:
                    ws.stuck_counter[aid] += 1

            ws.agents_at_exit = {
                a for a, p in ws.agent_positions.items() if p == ws.exit_position
            }

            if event_logger is not None:
                event_logger.log_step(
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

            if console_logger is not None:
                console_logger.log_step(ws, agent, record)

            if _check_end(ws, turn_limit):
                break

        if ws.status != "running":
            break
        ws.turn += 1

    if console_logger is not None:
        console_logger.log_end(ws)
    return ws
