"""Unit tests for ``classify_failure`` covering each of the four categories.

Each branch of the classifier docstring is exercised at least once. The
parametrized smoke case at the bottom guards against silent regressions in
the divergence ↔ classification pipeline by re-running every case through
the full ``compute_divergences`` → ``classify_failure`` chain.
"""

from __future__ import annotations

from typing import Any

import pytest

from dungeon.events import classify_failure, compute_divergences


def test_success_returns_none(belief_factory, truth_factory) -> None:
    belief = belief_factory()
    truth = truth_factory()
    result = {"ok": True, "moved": True, "position": [3, 4]}
    divergences = compute_divergences(belief, truth, provenance={}, now_turn=10)
    cls = classify_failure(
        tool_name="move",
        tool_input={"direction": "south"},
        result=result,
        belief_snapshot=belief,
        divergences=divergences,
    )
    assert cls is None


def test_environment_constraint_unseen_wall(belief_factory) -> None:
    """Move into a cell the agent had never observed; classifier excuses it."""
    belief = belief_factory(position=(3, 3), seen=[(3, 3), (2, 3), (4, 3)])
    result = {"ok": True, "moved": False, "note": "you didn't move", "blocked_by": "wall"}
    cls = classify_failure(
        tool_name="move",
        tool_input={"direction": "south"},
        result=result,
        belief_snapshot=belief,
        divergences=[],
    )
    assert cls == "environment_constraint"


def test_agent_error_seen_wall(belief_factory) -> None:
    """Move into a wall the agent had previously seen → the agent is at fault."""
    belief = belief_factory(position=(3, 3), seen=[(3, 3), (3, 4)])
    result = {"ok": True, "moved": False, "note": "you didn't move", "blocked_by": "wall"}
    cls = classify_failure(
        tool_name="move",
        tool_input={"direction": "south"},
        result=result,
        belief_snapshot=belief,
        divergences=[],
    )
    assert cls == "agent_error"


def test_coordination_failure_stale_partner(belief_factory, truth_factory) -> None:
    """A's belief about B is stale because B itself moved → coordination_failure."""
    belief = belief_factory(
        self_id="A",
        position=(3, 3),
        seen=[(3, 3), (3, 4)],
        other_pos=(5, 5),  # stale
        facts_last_seen={"other_position": 7},
    )
    result = {"ok": True, "moved": False, "note": "you didn't move", "blocked_by": "other_agent"}
    provenance = {"agent_B_position": {"agent": "B", "turn": 9, "change": "moved"}}
    truth = truth_factory(agent_positions={"A": [3, 3], "B": [3, 4]})
    divergences = compute_divergences(belief, truth, provenance, now_turn=10)
    cls = classify_failure(
        tool_name="move",
        tool_input={"direction": "south"},
        result=result,
        belief_snapshot=belief,
        divergences=divergences,
    )
    assert cls == "coordination_failure"


def test_information_lag_self_invalidated(belief_factory, truth_factory) -> None:
    """A picked the key up earlier and forgot → information_lag, not coordination."""
    belief = belief_factory(
        self_id="A",
        position=(0, 4),
        seen=[(0, 4)],
        key_pos=(0, 4),
        facts_last_seen={"key_position": 5},
    )
    result = {"ok": True, "success": False, "reason": "no key in this cell", "inventory": []}
    provenance = {"key_position": {"agent": "A", "turn": 8, "change": "picked_up"}}
    truth = truth_factory(
        key_position=None, key_holder=None, agent_positions={"A": [0, 4], "B": [7, 7]}
    )
    divergences = compute_divergences(belief, truth, provenance, now_turn=10)
    cls = classify_failure(
        tool_name="pick_up",
        tool_input={"item": "key"},
        result=result,
        belief_snapshot=belief,
        divergences=divergences,
    )
    assert cls == "information_lag"


def test_coordination_failure_partner_took_key(belief_factory, truth_factory) -> None:
    """B picked up the key after A last observed it → coordination_failure."""
    belief = belief_factory(
        self_id="A",
        position=(0, 4),
        seen=[(0, 4)],
        key_pos=(0, 4),
        facts_last_seen={"key_position": 3},
    )
    result = {"ok": True, "success": False, "reason": "no key in this cell", "inventory": []}
    provenance = {"key_position": {"agent": "B", "turn": 7, "change": "picked_up"}}
    truth = truth_factory(
        key_position=None, key_holder="B", agent_positions={"A": [0, 4], "B": [4, 4]}
    )
    divergences = compute_divergences(belief, truth, provenance, now_turn=10)
    cls = classify_failure(
        tool_name="pick_up",
        tool_input={"item": "key"},
        result=result,
        belief_snapshot=belief,
        divergences=divergences,
    )
    assert cls == "coordination_failure"


# ---------------------------------------------------------------------------
# Parametrized regression net: one row per documented classifier branch.
# The `_runner` helper makes it impossible for a future change to
# accidentally short-circuit a branch without a corresponding row failing.
# ---------------------------------------------------------------------------


def _runner(
    belief: dict[str, Any],
    truth: dict[str, Any],
    provenance: dict[str, Any],
    tool_name: str,
    tool_input: dict[str, Any],
    result: dict[str, Any],
) -> str | None:
    divergences = compute_divergences(belief, truth, provenance, now_turn=10)
    return classify_failure(
        tool_name=tool_name,
        tool_input=tool_input,
        result=result,
        belief_snapshot=belief,
        divergences=divergences,
    )


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("success", None),
        ("env_constraint", "environment_constraint"),
        ("agent_error", "agent_error"),
        ("coord_partner_moved", "coordination_failure"),
        ("info_lag_self", "information_lag"),
        ("coord_partner_took_key", "coordination_failure"),
    ],
)
def test_classifier_branch_coverage(
    scenario: str,
    expected: str | None,
    belief_factory,
    truth_factory,
) -> None:
    if scenario == "success":
        belief = belief_factory()
        truth = truth_factory()
        result = {"ok": True, "moved": True, "position": [3, 4]}
        cls = _runner(belief, truth, {}, "move", {"direction": "south"}, result)
    elif scenario == "env_constraint":
        belief = belief_factory(position=(3, 3), seen=[(3, 3), (2, 3), (4, 3)])
        truth = truth_factory()
        result = {"ok": True, "moved": False, "blocked_by": "wall"}
        cls = _runner(belief, truth, {}, "move", {"direction": "south"}, result)
    elif scenario == "agent_error":
        belief = belief_factory(position=(3, 3), seen=[(3, 3), (3, 4)])
        truth = truth_factory()
        result = {"ok": True, "moved": False, "blocked_by": "wall"}
        cls = _runner(belief, truth, {}, "move", {"direction": "south"}, result)
    elif scenario == "coord_partner_moved":
        belief = belief_factory(
            self_id="A",
            position=(3, 3),
            seen=[(3, 3), (3, 4)],
            other_pos=(5, 5),
            facts_last_seen={"other_position": 7},
        )
        truth = truth_factory(agent_positions={"A": [3, 3], "B": [3, 4]})
        provenance = {"agent_B_position": {"agent": "B", "turn": 9, "change": "moved"}}
        result = {"ok": True, "moved": False, "blocked_by": "other_agent"}
        cls = _runner(belief, truth, provenance, "move", {"direction": "south"}, result)
    elif scenario == "info_lag_self":
        belief = belief_factory(
            self_id="A",
            position=(0, 4),
            seen=[(0, 4)],
            key_pos=(0, 4),
            facts_last_seen={"key_position": 5},
        )
        truth = truth_factory(
            key_position=None, key_holder=None, agent_positions={"A": [0, 4], "B": [7, 7]}
        )
        provenance = {"key_position": {"agent": "A", "turn": 8, "change": "picked_up"}}
        result = {"ok": True, "success": False, "reason": "no key in this cell"}
        cls = _runner(belief, truth, provenance, "pick_up", {"item": "key"}, result)
    elif scenario == "coord_partner_took_key":
        belief = belief_factory(
            self_id="A",
            position=(0, 4),
            seen=[(0, 4)],
            key_pos=(0, 4),
            facts_last_seen={"key_position": 3},
        )
        truth = truth_factory(
            key_position=None, key_holder="B", agent_positions={"A": [0, 4], "B": [4, 4]}
        )
        provenance = {"key_position": {"agent": "B", "turn": 7, "change": "picked_up"}}
        result = {"ok": True, "success": False, "reason": "no key in this cell"}
        cls = _runner(belief, truth, provenance, "pick_up", {"item": "key"}, result)
    else:  # pragma: no cover - safety net
        pytest.fail(f"unknown scenario: {scenario}")

    assert cls == expected
