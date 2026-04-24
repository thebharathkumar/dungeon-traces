"""Property test for BeliefState invariants under arbitrary action sequences.

The classifier relies on a contract that BeliefState's ``position`` field
is always equal to the agent's true position because the move tool returns
the post-action position even on silent failure (so the agent self-position
belief cannot go stale). This file verifies that contract holds across
randomly drawn action sequences applied to a freshly generated world.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings, strategies as st

from dungeon.agent import DungeonAgent
from dungeon.tools import execute_tool
from dungeon.world import generate_world

_DIRECTIONS = st.sampled_from(["north", "south", "east", "west"])
_ACTION_SEQUENCE = st.lists(_DIRECTIONS, min_size=1, max_size=20)
_SEEDS = st.integers(min_value=0, max_value=999)


@given(seed=_SEEDS, actions=_ACTION_SEQUENCE)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_self_position_belief_is_never_stale(seed: int, actions: list[str]) -> None:
    """After any sequence of moves, agent.belief.position must equal truth.

    Anthropic's tool API forces tool_move to return the current position on
    every call (success or silent failure), so BeliefState.position is
    sourced directly from the tool result. The invariant must hold no
    matter what sequence of valid directions the model emits.
    """
    ws = generate_world(seed, ["A", "B"])
    agent_a = DungeonAgent(
        agent_id="A",
        client=None,  # tool execution path doesn't touch the LLM
        model="mock",
        starting_position=ws.agent_positions["A"],
    )
    for direction in actions:
        result = execute_tool(ws, "A", "move", {"direction": direction})
        agent_a.belief.update_from_tool_result(ws.turn, "move", {"direction": direction}, result)
        assert tuple(agent_a.belief.position) == ws.agent_positions["A"], (
            f"belief drift after move({direction}): "
            f"belief={agent_a.belief.position} truth={ws.agent_positions['A']}"
        )


@given(seed=_SEEDS, actions=_ACTION_SEQUENCE)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_seen_cells_grow_monotonically(seed: int, actions: list[str]) -> None:
    """The set of cells the agent has seen never shrinks across actions.

    Once a cell is observed it stays in seen_cells forever (the value may
    update, but the key persists). The classifier uses this set to decide
    environment_constraint vs agent_error; if it ever contracted, an
    agent_error could be misclassified as environment_constraint after a
    revisit.
    """
    ws = generate_world(seed, ["A", "B"])
    agent_a = DungeonAgent(
        agent_id="A",
        client=None,
        model="mock",
        starting_position=ws.agent_positions["A"],
    )
    previous: set[tuple[int, int]] = set()
    for direction in actions:
        result = execute_tool(ws, "A", "move", {"direction": direction})
        agent_a.belief.update_from_tool_result(ws.turn, "move", {"direction": direction}, result)
        current = set(agent_a.belief.seen_cells.keys())
        assert previous.issubset(current), (
            f"seen_cells shrank: lost {previous - current} after move({direction})"
        )
        previous = current
