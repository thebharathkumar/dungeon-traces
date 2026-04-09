"""Game loop: turn scheduling, message delivery, end conditions.

Turn model: one integer `ws.turn` per full round. Both agents act on the
same turn number in strict A-then-B order. Messages sent on turn N are
delivered at the START of turn N+1, before either agent acts. This matches
the spec's "delivered on the following turn, not instantly" rule and means
the delay is experienced even if B acts directly after A on the same turn N.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .agent import DungeonAgent
from .world import DEFAULT_TURN_LIMIT, STUCK_THRESHOLD, WorldState, generate_world


class StepLogger(Protocol):
    def log_step(self, ws: WorldState, agent: DungeonAgent, record: dict) -> None: ...
    def log_end(self, ws: WorldState) -> None: ...


@dataclass
class ConsoleLogger:
    """Minimal stdout printer used during Phase 1 development.

    Phase 2 will introduce a richer event logger and keep this one around
    for human-readable local debugging.
    """

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
    turn_limit: int = DEFAULT_TURN_LIMIT,
    logger: Optional[StepLogger] = None,
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
            record = agent.take_turn(ws)

            # Stuck counter: consecutive failed moves only. Any successful
            # move resets it. Other tools leave it alone.
            if record.get("tool_name") == "move":
                if record.get("semantic_success"):
                    ws.stuck_counter[aid] = 0
                else:
                    ws.stuck_counter[aid] += 1

            # Recompute the "currently at exit" set from ground truth.
            ws.agents_at_exit = {
                a for a, p in ws.agent_positions.items() if p == ws.exit_position
            }

            if logger is not None:
                logger.log_step(ws, agent, record)

            if _check_end(ws, turn_limit):
                break

        if ws.status != "running":
            break
        ws.turn += 1

    if logger is not None:
        logger.log_end(ws)
    return ws
