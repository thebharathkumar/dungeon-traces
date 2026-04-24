"""LLM-backed dungeon agent and its private belief state.

The agent is intentionally stateless across turns at the LLM layer: each turn
is a single fresh completion with no conversation history. Memory lives in
BeliefState, which is rendered into the prompt as a compact summary. This
keeps traces per-turn clean and makes it trivial for Phase 2 to diff belief
against ground truth at a single point in time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .llm import call_with_retry
from .tools import DIRECTION_DELTAS, TOOL_SCHEMAS, execute_tool, is_semantic_success
from .world import WorldState

Pos = tuple[int, int]

SYSTEM_PROMPT_TEMPLATE = (
    "You are {agent_id}, one of two agents exploring a dungeon. Your shared "
    "goal is for BOTH agents to reach the exit cell. One of you must find and "
    "pick up the key, then use_item(item='key', target='door') on the locked "
    "door. The door must be unlocked before anyone can step onto the exit.\n"
    "\n"
    "You can only see the 3x3 cells around your current position. To "
    "explore, you MUST call move. The move tool returns the new 3x3 "
    "visibility automatically, so observe is only useful once at the start "
    "or after read_messages/pick_up/use_item. Do NOT call observe two turns "
    "in a row. When in doubt, call move in the direction of the nearest "
    "unknown area, the key, or the exit. Messages you send are delivered "
    "on the NEXT turn, not instantly, so the other agent may act on old "
    "information.\n"
    "\n"
    "Think in ONE short sentence, then call exactly one tool. Do not call "
    "multiple tools in a single turn."
)


@dataclass
class BeliefState:
    """Everything an agent thinks is true about the world.

    Updated only from tool return values, never from ground truth directly.
    The point of keeping this separate from WorldState is so that Phase 2
    can diff the two and surface stale beliefs.
    """

    self_id: str
    position: Pos
    inventory: list[str] = field(default_factory=list)
    # (x,y) -> {type, contents, turn_seen}
    seen_cells: dict[Pos, dict] = field(default_factory=dict)
    # "unknown" | (x,y) | "in_my_inventory" | "in_partner_inventory"
    last_known_key_position: Any = "unknown"
    # "unknown" | True | False
    last_known_door_locked: Any = "unknown"
    # "unknown" | (x,y)
    last_known_other_position: Any = "unknown"
    # fact_key -> turn last confirmed
    facts_last_seen: dict[str, int] = field(default_factory=dict)
    # cumulative log of messages the agent has actually read
    inbox_history: list[dict] = field(default_factory=list)

    def update_from_visible(self, visible_cells: list[dict], turn: int) -> None:
        for cell in visible_cells:
            pos = (cell["x"], cell["y"])
            self.seen_cells[pos] = {**cell, "turn_seen": turn}
            contents = cell.get("contents", [])

            if "key" in contents:
                self.last_known_key_position = pos
                self.facts_last_seen["key_position"] = turn
            if "door_locked" in contents:
                self.last_known_door_locked = True
                self.facts_last_seen["door_locked"] = turn
            if "door_open" in contents:
                self.last_known_door_locked = False
                self.facts_last_seen["door_locked"] = turn

            for c in contents:
                if c.startswith("agent:"):
                    other_id = c.split(":", 1)[1]
                    if other_id != self.self_id:
                        self.last_known_other_position = pos
                        self.facts_last_seen["other_position"] = turn

    def update_from_tool_result(
        self,
        turn: int,
        tool_name: str,
        tool_input: dict,
        result: dict,
    ) -> None:
        if tool_name == "move":
            # position always reflects reality (tool returns current position
            # even on silent failure) so the agent self-position belief cannot
            # actually go stale. This is intentional.
            if "position" in result:
                self.position = tuple(result["position"])
            self.update_from_visible(result.get("visible", []), turn)

        elif tool_name == "observe":
            if "position" in result:
                self.position = tuple(result["position"])
            if "inventory" in result:
                self.inventory = list(result["inventory"])
            self.update_from_visible(result.get("visible", []), turn)

        elif tool_name == "pick_up":
            if result.get("success"):
                self.inventory = list(result.get("inventory", []))
                if tool_input.get("item") == "key":
                    self.last_known_key_position = "in_my_inventory"
                    self.facts_last_seen["key_position"] = turn

        elif tool_name == "use_item":
            if result.get("success"):
                if tool_input.get("item") == "key" and tool_input.get("target") == "door":
                    self.last_known_door_locked = False
                    self.facts_last_seen["door_locked"] = turn

        elif tool_name == "read_messages":
            for m in result.get("messages", []):
                self.inbox_history.append(m)

    def snapshot(self) -> dict:
        """Flat, JSON-ready belief snapshot for event logging.

        seen_cell_positions is included as a list so the classifier (and
        Phase 3 viewer) can tell "the agent had observed this cell before"
        without needing the full BeliefState object.
        """
        return {
            "self_id": self.self_id,
            "position": list(self.position),
            "inventory": list(self.inventory),
            "last_known_key_position": _jsonify(self.last_known_key_position),
            "last_known_door_locked": self.last_known_door_locked,
            "last_known_other_position": _jsonify(self.last_known_other_position),
            "facts_last_seen": dict(self.facts_last_seen),
            "seen_cell_count": len(self.seen_cells),
            "seen_cell_positions": sorted([list(p) for p in self.seen_cells.keys()]),
        }

    def render_for_prompt(self, turn: int) -> str:
        lines = [
            f"Current turn: {turn}",
            f"Your id: {self.self_id}",
            f"Your position: ({self.position[0]}, {self.position[1]})",
            f"Your inventory: {self.inventory or 'empty'}",
            "",
            "What you believe about the world (may be stale):",
            f"  key: {_describe_fact(self.last_known_key_position, self.facts_last_seen.get('key_position'), turn)}",
            f"  door locked: {_describe_fact(self.last_known_door_locked, self.facts_last_seen.get('door_locked'), turn)}",
            f"  partner position: {_describe_fact(self.last_known_other_position, self.facts_last_seen.get('other_position'), turn)}",
            "",
            f"Cells you have seen so far: {len(self.seen_cells)}",
        ]
        if self.inbox_history:
            lines.append("")
            lines.append("Messages you have already read (most recent last):")
            for m in self.inbox_history[-5:]:
                lines.append(f"  [turn {m.get('sent_turn')}] {m.get('from')}: {m.get('content')}")
        return "\n".join(lines)


def _describe_fact(value: Any, last_seen_turn: Optional[int], now_turn: int) -> str:
    if value == "unknown" or value is None:
        return "unknown (never observed)"
    if last_seen_turn is None:
        return f"{value}"
    age = now_turn - last_seen_turn
    if age == 0:
        return f"{_jsonify(value)} (observed this turn)"
    return f"{_jsonify(value)} (observed {age} turn{'s' if age != 1 else ''} ago)"


def _jsonify(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


class DungeonAgent:
    """Wraps an LLM client plus the per-agent belief state.

    One instance per agent. The simulation game loop calls take_turn() on each
    agent in strict alternation. take_turn makes exactly one LLM call, executes
    exactly one tool, and returns a structured record that the game loop hands
    to the Phase 2 event logger.
    """

    def __init__(
        self,
        agent_id: str,
        client: Any,
        model: str,
        starting_position: Pos,
    ) -> None:
        self.agent_id = agent_id
        self.client = client
        self.model = model
        self.belief = BeliefState(self_id=agent_id, position=starting_position)
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(agent_id=agent_id)
        # Compact one-line feedback about the previous turn's tool call,
        # rendered into the next user prompt so the model sees that (say) its
        # last move bumped a wall. Cleared to None on success since the belief
        # snapshot already reflects the new position.
        self.last_action_feedback: Optional[str] = None

    def _build_user_prompt(self, ws: WorldState) -> str:
        unread = len(ws.inboxes[self.agent_id])
        unread_line = (
            f"You have {unread} unread message(s) in your inbox. "
            "Call read_messages to see them."
            if unread > 0
            else "You have no unread messages."
        )
        feedback_block = (
            f"\n{self.last_action_feedback}\n" if self.last_action_feedback else ""
        )
        return (
            f"{self.belief.render_for_prompt(ws.turn)}\n"
            f"{feedback_block}"
            f"\n"
            f"{unread_line}\n"
            f"\n"
            f"Think in one short sentence, then call exactly one tool."
        )

    def take_turn(self, ws: WorldState) -> dict:
        user_prompt = self._build_user_prompt(ws)
        t0 = time.perf_counter()
        response = call_with_retry(
            self.client,
            model=self.model,
            max_tokens=512,
            system=self.system_prompt,
            tools=TOOL_SCHEMAS,
            messages=[{"role": "user", "content": user_prompt}],
        )
        llm_latency_ms = int((time.perf_counter() - t0) * 1000)

        reasoning = ""
        tool_name: Optional[str] = None
        tool_input: dict = {}
        serialized_blocks: list[dict] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                reasoning += block.text
                serialized_blocks.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                block_input = dict(block.input)
                serialized_blocks.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", None),
                        "name": block.name,
                        "input": block_input,
                    }
                )
                if tool_name is None:
                    tool_name = block.name
                    tool_input = block_input

        tool_t0 = time.perf_counter()
        if tool_name is None:
            result = {
                "ok": False,
                "reason": "model did not call a tool",
                "success": False,
            }
        else:
            result = execute_tool(ws, self.agent_id, tool_name, tool_input)
            self.belief.update_from_tool_result(ws.turn, tool_name, tool_input, result)
        tool_latency_ms = int((time.perf_counter() - tool_t0) * 1000)

        semantic_success = is_semantic_success(tool_name or "", result)
        self.last_action_feedback = _format_last_action_feedback(
            tool_name, tool_input, result, semantic_success, self.belief.position
        )

        usage = getattr(response, "usage", None)
        return {
            "agent_id": self.agent_id,
            "turn": ws.turn,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "reasoning": reasoning.strip(),
            "response_content_blocks": serialized_blocks,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_result": result,
            "semantic_success": semantic_success,
            "belief_snapshot": self.belief.snapshot(),
            "llm_latency_ms": llm_latency_ms,
            "tool_latency_ms": tool_latency_ms,
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            } if usage else None,
        }


def _format_last_action_feedback(
    tool_name: Optional[str],
    tool_input: dict,
    result: dict,
    semantic_success: bool,
    current_pos: Pos,
) -> Optional[str]:
    """Compact one-line feedback to thread into the next turn's prompt.

    Only emitted on failure. Successes are already reflected in the updated
    belief snapshot, so echoing them back would just waste tokens.
    """
    if semantic_success:
        return None
    if tool_name is None:
        reason = result.get("reason") or "no tool call produced"
        return f"Last action: (no tool call) -> FAILED: {reason}"

    if tool_name == "move":
        direction = tool_input.get("direction", "?")
        blocked_by = result.get("blocked_by") or "unknown"
        target = None
        if direction in DIRECTION_DELTAS:
            dx, dy = DIRECTION_DELTAS[direction]
            target = (current_pos[0] + dx, current_pos[1] + dy)
        target_str = f" at ({target[0]}, {target[1]})" if target is not None else ""
        return (
            f"Last action: move(direction='{direction}') -> FAILED: "
            f"blocked by {blocked_by}{target_str}. Try a different direction."
        )

    if tool_name == "pick_up":
        item = tool_input.get("item", "?")
        reason = result.get("reason") or "no reason given"
        return f"Last action: pick_up(item='{item}') -> FAILED: {reason}"

    if tool_name == "use_item":
        item = tool_input.get("item", "?")
        target = tool_input.get("target", "?")
        reason = result.get("reason") or "no reason given"
        return (
            f"Last action: use_item(item='{item}', target='{target}') "
            f"-> FAILED: {reason}"
        )

    reason = result.get("reason") or result.get("note") or "unknown failure"
    return f"Last action: {tool_name}(...) -> FAILED: {reason}"
