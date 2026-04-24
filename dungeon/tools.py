"""Agent tools.

Each tool takes a WorldState plus the calling agent id and returns a dict
that is both (a) sent back to the LLM as the tool result and (b) consumed
by the Phase 2 event logger. The shapes are therefore intentionally simple
and stable across tools: every result has an "ok" boolean; semantic success
(did the pickup actually work?) is expressed in a per-tool field.

Silent failure: move() returning {moved: false} is intentional per the spec.
A wall collision is NOT an error; the agent is expected to infer failure from
the fact that its position did not change. This is what produces the "agent
believes it moved but it did not" class of trace bugs.
"""

from __future__ import annotations

from typing import Any

from .world import Message, WorldState

DIRECTION_DELTAS = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


TOOL_SCHEMAS: list[dict] = [
    {
        "name": "move",
        "description": (
            "Move one cell in a cardinal direction. If the destination is a wall, "
            "the edge of the map, a locked door, or occupied, you will not move. "
            "You will still receive your current position and a 3x3 view."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["north", "south", "east", "west"],
                    "description": "Cardinal direction to move one cell.",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "observe",
        "description": "Return your current position, inventory, and the 3x3 cells around you.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "pick_up",
        "description": "Pick up an item in your current cell. Fails if no such item is here.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": "Name of the item to pick up, e.g. 'key'.",
                },
            },
            "required": ["item"],
        },
    },
    {
        "name": "use_item",
        "description": (
            "Use an item from your inventory on a target. For example, "
            "use_item(item='key', target='door') unlocks an adjacent door."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["item", "target"],
        },
    },
    {
        "name": "send_message",
        "description": (
            "Queue a message to the other agent. Messages are delivered on the "
            "following turn, not instantly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "read_messages",
        "description": "Read all messages waiting in your inbox. The inbox is cleared after reading.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _other_agent(ws: WorldState, agent_id: str) -> str | None:
    for aid in ws.agent_positions:
        if aid != agent_id:
            return aid
    return None


def tool_move(ws: WorldState, agent_id: str, direction: str) -> dict:
    if direction not in DIRECTION_DELTAS:
        return {
            "ok": False,
            "moved": False,
            "reason": f"invalid direction '{direction}'",
        }
    cur = ws.agent_positions[agent_id]
    dx, dy = DIRECTION_DELTAS[direction]
    new_pos = (cur[0] + dx, cur[1] + dy)

    blocked_reason = None
    if not ws.in_bounds(new_pos):
        blocked_reason = "edge"
    elif ws.cell_type(new_pos).value == "wall":
        blocked_reason = "wall"
    elif new_pos == ws.door_position and ws.door_locked:
        blocked_reason = "locked_door"
    elif ws.agent_at(new_pos) is not None:
        blocked_reason = "other_agent"

    if blocked_reason is not None:
        # Silent failure on purpose: the tool succeeds, the move does not.
        # blocked_by is included so the Phase 2 classifier can tell a wall
        # collision apart from a locked-door or other-agent collision. The
        # LLM never sees this dict because we do not send tool_result back
        # into the model conversation.
        return {
            "ok": True,
            "moved": False,
            "position": list(cur),
            "visible": ws.visible_cells(cur),
            "note": "you didn't move",
            "blocked_by": blocked_reason,
        }

    ws.agent_positions[agent_id] = new_pos
    if new_pos == ws.exit_position:
        ws.agents_at_exit.add(agent_id)
    ws.record_provenance(f"agent_{agent_id}_position", agent_id, "moved")
    return {
        "ok": True,
        "moved": True,
        "position": list(new_pos),
        "visible": ws.visible_cells(new_pos),
    }


def tool_observe(ws: WorldState, agent_id: str) -> dict:
    pos = ws.agent_positions[agent_id]
    return {
        "ok": True,
        "position": list(pos),
        "inventory": sorted(ws.agent_inventories[agent_id]),
        "visible": ws.visible_cells(pos),
    }


def tool_pick_up(ws: WorldState, agent_id: str, item: str) -> dict:
    pos = ws.agent_positions[agent_id]
    inv = ws.agent_inventories[agent_id]
    if item == "key":
        if ws.key_position == pos:
            inv.add("key")
            ws.key_position = None
            ws.key_holder = agent_id
            ws.record_provenance("key_position", agent_id, "picked_up")
            return {
                "ok": True,
                "success": True,
                "picked_up": "key",
                "inventory": sorted(inv),
            }
        return {
            "ok": True,
            "success": False,
            "reason": "no key in this cell",
            "inventory": sorted(inv),
        }
    return {
        "ok": True,
        "success": False,
        "reason": f"unknown or unavailable item '{item}'",
        "inventory": sorted(inv),
    }


def tool_use_item(ws: WorldState, agent_id: str, item: str, target: str) -> dict:
    pos = ws.agent_positions[agent_id]
    inv = ws.agent_inventories[agent_id]
    if item not in inv:
        return {"ok": True, "success": False, "reason": f"you do not have a {item}"}
    if item == "key" and target == "door":
        dist = max(
            abs(pos[0] - ws.door_position[0]),
            abs(pos[1] - ws.door_position[1]),
        )
        if dist > 1:
            return {"ok": True, "success": False, "reason": "no door adjacent to you"}
        if not ws.door_locked:
            return {"ok": True, "success": False, "reason": "door is already unlocked"}
        ws.door_locked = False
        ws.record_provenance("door_locked", agent_id, "unlocked")
        return {"ok": True, "success": True, "note": "door unlocked"}
    return {"ok": True, "success": False, "reason": f"cannot use {item} on {target}"}


def tool_send_message(ws: WorldState, agent_id: str, content: str) -> dict:
    recipient = _other_agent(ws, agent_id)
    if recipient is None:
        return {"ok": False, "reason": "no other agent to message"}
    msg = Message(
        sender=agent_id,
        recipient=recipient,
        content=content,
        sent_turn=ws.turn,
        deliver_turn=ws.turn + 1,
    )
    ws.pending_messages.append(msg)
    return {
        "ok": True,
        "queued": True,
        "recipient": recipient,
        "will_deliver_turn": ws.turn + 1,
    }


def tool_read_messages(ws: WorldState, agent_id: str) -> dict:
    msgs = ws.inboxes[agent_id]
    serialized = [
        {
            "from": m.sender,
            "content": m.content,
            "sent_turn": m.sent_turn,
            "delivered_turn": ws.turn,
        }
        for m in msgs
    ]
    ws.inboxes[agent_id] = []
    return {"ok": True, "count": len(serialized), "messages": serialized}


def execute_tool(
    ws: WorldState,
    agent_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
) -> dict:
    if tool_name == "move":
        return tool_move(ws, agent_id, tool_input.get("direction", ""))
    if tool_name == "observe":
        return tool_observe(ws, agent_id)
    if tool_name == "pick_up":
        return tool_pick_up(ws, agent_id, tool_input.get("item", ""))
    if tool_name == "use_item":
        return tool_use_item(
            ws,
            agent_id,
            tool_input.get("item", ""),
            tool_input.get("target", ""),
        )
    if tool_name == "send_message":
        return tool_send_message(ws, agent_id, tool_input.get("content", ""))
    if tool_name == "read_messages":
        return tool_read_messages(ws, agent_id)
    return {"ok": False, "reason": f"unknown tool '{tool_name}'"}


def is_semantic_success(tool_name: str, result: dict) -> bool:
    """True if the tool did the thing it was asked to do.

    Phase 2 event logger uses this to distinguish agent no-ops (moved:false)
    from genuine successes, without every call site needing to know the shape.
    """
    if not result.get("ok", False):
        return False
    if tool_name == "move":
        return bool(result.get("moved"))
    if tool_name in ("pick_up", "use_item"):
        return bool(result.get("success"))
    if tool_name in ("observe", "read_messages", "send_message"):
        return True
    return False
