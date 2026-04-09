"""Targeted test of classify_failure covering each of the four categories.

Not a full pytest suite, just assertions that exercise each branch so a
reviewer can read it top-to-bottom and see that the classifier behaves
the way the docstring claims. Run with:
    python scripts/test_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dungeon.events import classify_failure, compute_divergences


def _belief(self_id="A", position=(3, 3), seen=None, key_pos="unknown", door_locked="unknown", other_pos="unknown", facts_last_seen=None):
    return {
        "self_id": self_id,
        "position": list(position),
        "inventory": [],
        "last_known_key_position": key_pos if key_pos == "unknown" else list(key_pos) if isinstance(key_pos, tuple) else key_pos,
        "last_known_door_locked": door_locked,
        "last_known_other_position": other_pos if other_pos == "unknown" else list(other_pos),
        "facts_last_seen": facts_last_seen or {},
        "seen_cell_count": len(seen or []),
        "seen_cell_positions": sorted([list(p) for p in (seen or [])]),
    }


def _truth(key_position=None, key_holder=None, door_locked=True, agent_positions=None):
    return {
        "turn": 10,
        "agent_positions": agent_positions or {"A": [3, 3], "B": [7, 7]},
        "agent_inventories": {"A": [], "B": []},
        "key_position": list(key_position) if key_position else None,
        "key_holder": key_holder,
        "door_position": [6, 7],
        "door_locked": door_locked,
        "exit_position": [7, 7],
        "agents_at_exit": [],
        "status": "running",
    }


def case_success() -> None:
    belief = _belief()
    truth = _truth()
    result = {"ok": True, "moved": True, "position": [3, 4]}
    divergences = compute_divergences(belief, truth, provenance={}, now_turn=10)
    cls = classify_failure(
        tool_name="move",
        tool_input={"direction": "south"},
        result=result,
        belief_snapshot=belief,
        divergences=divergences,
    )
    assert cls is None, f"expected None, got {cls}"
    print("success -> None ok")


def case_environment_constraint() -> None:
    # Agent moved into a cell it had never observed before, hit a wall.
    belief = _belief(position=(3, 3), seen=[(3, 3), (2, 3), (4, 3)])
    result = {"ok": True, "moved": False, "note": "you didn't move", "blocked_by": "wall"}
    divergences: list[dict] = []
    cls = classify_failure(
        tool_name="move",
        tool_input={"direction": "south"},
        result=result,
        belief_snapshot=belief,
        divergences=divergences,
    )
    assert cls == "environment_constraint", f"expected environment_constraint, got {cls}"
    print("unseen wall -> environment_constraint ok")


def case_agent_error_seen_wall() -> None:
    # Agent moved into a wall it had previously seen.
    belief = _belief(position=(3, 3), seen=[(3, 3), (3, 4)])
    result = {"ok": True, "moved": False, "note": "you didn't move", "blocked_by": "wall"}
    cls = classify_failure(
        tool_name="move",
        tool_input={"direction": "south"},
        result=result,
        belief_snapshot=belief,
        divergences=[],
    )
    assert cls == "agent_error", f"expected agent_error, got {cls}"
    print("seen wall -> agent_error ok")


def case_coordination_failure_stale_partner() -> None:
    # A thinks B is at (3,4); A moves south to (3,4); B is actually there still.
    belief = _belief(
        self_id="A",
        position=(3, 3),
        seen=[(3, 3), (3, 4)],
        other_pos=(3, 4),
        facts_last_seen={"other_position": 7},
    )
    result = {"ok": True, "moved": False, "note": "you didn't move", "blocked_by": "other_agent"}
    # Provenance says B last moved at turn 9 (more recent than last_seen at turn 7)
    provenance = {"agent_B_position": {"agent": "B", "turn": 9, "change": "moved"}}
    truth = _truth(agent_positions={"A": [3, 3], "B": [3, 4]})
    # tweak belief to a stale position
    belief["last_known_other_position"] = [5, 5]
    divergences = compute_divergences(belief, truth, provenance, now_turn=10)
    cls = classify_failure(
        tool_name="move",
        tool_input={"direction": "south"},
        result=result,
        belief_snapshot=belief,
        divergences=divergences,
    )
    assert cls == "coordination_failure", f"expected coordination_failure, got {cls}"
    print("stale partner caused by B -> coordination_failure ok")


def case_information_lag_pickup_stale_key() -> None:
    # A believed key was at (0,4); A walked to (0,4) and tries pick_up; key is gone
    # because A itself removed it earlier (provenance self). This is information_lag,
    # not coordination_failure, because no other agent invalidated the belief.
    belief = _belief(
        self_id="A",
        position=(0, 4),
        seen=[(0, 4)],
        key_pos=(0, 4),
        facts_last_seen={"key_position": 5},
    )
    result = {"ok": True, "success": False, "reason": "no key in this cell", "inventory": []}
    provenance = {"key_position": {"agent": "A", "turn": 8, "change": "picked_up"}}
    truth = _truth(key_position=None, key_holder=None, agent_positions={"A": [0, 4], "B": [7, 7]})
    divergences = compute_divergences(belief, truth, provenance, now_turn=10)
    cls = classify_failure(
        tool_name="pick_up",
        tool_input={"item": "key"},
        result=result,
        belief_snapshot=belief,
        divergences=divergences,
    )
    assert cls == "information_lag", f"expected information_lag, got {cls}"
    print("stale key caused by self -> information_lag ok")


def case_coordination_failure_partner_took_key() -> None:
    # A saw key at (0,4) at turn 3. B picked it up at turn 7. A tries to pick up at turn 10.
    belief = _belief(
        self_id="A",
        position=(0, 4),
        seen=[(0, 4)],
        key_pos=(0, 4),
        facts_last_seen={"key_position": 3},
    )
    result = {"ok": True, "success": False, "reason": "no key in this cell", "inventory": []}
    provenance = {"key_position": {"agent": "B", "turn": 7, "change": "picked_up"}}
    truth = _truth(key_position=None, key_holder="B", agent_positions={"A": [0, 4], "B": [4, 4]})
    divergences = compute_divergences(belief, truth, provenance, now_turn=10)
    cls = classify_failure(
        tool_name="pick_up",
        tool_input={"item": "key"},
        result=result,
        belief_snapshot=belief,
        divergences=divergences,
    )
    assert cls == "coordination_failure", f"expected coordination_failure, got {cls}"
    print("partner took key -> coordination_failure ok")


def main() -> int:
    case_success()
    case_environment_constraint()
    case_agent_error_seen_wall()
    case_coordination_failure_stale_partner()
    case_information_lag_pickup_stale_key()
    case_coordination_failure_partner_took_key()
    print("\nall classifier cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
