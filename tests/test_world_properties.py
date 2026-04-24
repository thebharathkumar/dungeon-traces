"""Hypothesis-driven property tests for world generation.

Example-based tests answer "does this one seed work?" Property tests answer
"does the *invariant* hold across hundreds of seeds?" The invariants here
are ones every playable world must satisfy; if Hypothesis finds a seed that
violates one, ``generate_world`` has a bug regardless of how rare the case.

Strategy: Hypothesis draws random seeds in a bounded range and feeds them
to ``generate_world``. The properties below are all O(grid^2) checks.
"""

from __future__ import annotations

from collections import deque

from hypothesis import given, settings, strategies as st

from dungeon.world import (
    DEFAULT_TURN_LIMIT,
    GRID_SIZE,
    CellType,
    WorldState,
    generate_world,
)

# Bounded seed range keeps the test fast without sacrificing coverage:
# generate_world is fully deterministic on seed, so 0..9999 is plenty of
# distinct worlds and Hypothesis can shrink failures cleanly.
_SEEDS = st.integers(min_value=0, max_value=9999)
_AGENT_IDS = st.sampled_from([["A", "B"], ["X", "Y"], ["alpha", "beta"]])

# Hypothesis settings:
#   max_examples=200 is enough to surface most layout bugs while keeping
#   the whole suite well under a second on CI.
#   deadline=None disables the per-example timeout because cold-start the
#   first example sometimes exceeds 200ms.
_PROPERTY_SETTINGS = settings(max_examples=200, deadline=None)


def _bfs_reachable(ws: WorldState, start: tuple[int, int]) -> set[tuple[int, int]]:
    """BFS treating walls as impassable but the door cell as passable."""
    seen: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            n = (nx, ny)
            if n in seen:
                continue
            if ws.grid[ny][nx] == CellType.EMPTY or n == ws.door_position:
                seen.add(n)
                queue.append(n)
    return seen


@given(seed=_SEEDS, agent_ids=_AGENT_IDS)
@_PROPERTY_SETTINGS
def test_world_is_deterministic_for_seed(seed: int, agent_ids: list[str]) -> None:
    """Same seed + same agent_ids must produce byte-identical worlds.

    Determinism is the foundation of reproducible traces. If this property
    breaks, every regression run downstream becomes meaningless.
    """
    a = generate_world(seed, agent_ids)
    b = generate_world(seed, agent_ids)
    assert a.grid == b.grid
    assert a.key_position == b.key_position
    assert a.door_position == b.door_position
    assert a.exit_position == b.exit_position
    assert a.agent_positions == b.agent_positions


@given(seed=_SEEDS, agent_ids=_AGENT_IDS)
@_PROPERTY_SETTINGS
def test_critical_cells_are_reachable(seed: int, agent_ids: list[str]) -> None:
    """Key, door, exit, and every agent must be reachable from any spawn.

    BFS treats the door as passable so the property checks the *graph*
    after the door is unlocked. ``_try_generate`` enforces this internally;
    this test verifies it externally so a future refactor cannot silently
    weaken the constraint.
    """
    ws = generate_world(seed, agent_ids)
    start = next(iter(ws.agent_positions.values()))
    reachable = _bfs_reachable(ws, start)
    assert ws.key_position in reachable
    assert ws.door_position in reachable
    assert ws.exit_position in reachable
    for pos in ws.agent_positions.values():
        assert pos in reachable


@given(seed=_SEEDS, agent_ids=_AGENT_IDS)
@_PROPERTY_SETTINGS
def test_no_entity_is_on_a_wall(seed: int, agent_ids: list[str]) -> None:
    """Walking entities (agents, key) must spawn on EMPTY cells."""
    ws = generate_world(seed, agent_ids)
    assert ws.cell_type(ws.key_position) == CellType.EMPTY
    for aid, pos in ws.agent_positions.items():
        assert ws.cell_type(pos) == CellType.EMPTY, f"agent {aid} on wall at {pos}"


@given(seed=_SEEDS, agent_ids=_AGENT_IDS)
@_PROPERTY_SETTINGS
def test_no_entities_overlap_at_spawn(seed: int, agent_ids: list[str]) -> None:
    """No two agents share a cell, and no agent shares the key/door/exit."""
    ws = generate_world(seed, agent_ids)
    positions = list(ws.agent_positions.values())
    assert len(set(positions)) == len(positions), f"agent overlap: {positions}"
    forbidden = {ws.key_position, ws.door_position, ws.exit_position}
    for aid, pos in ws.agent_positions.items():
        assert pos not in forbidden, f"agent {aid} on {pos} (forbidden={forbidden})"


@given(seed=_SEEDS, agent_ids=_AGENT_IDS)
@_PROPERTY_SETTINGS
def test_door_is_the_only_path_to_exit(seed: int, agent_ids: list[str]) -> None:
    """With the door treated as a wall, the exit must be unreachable.

    The blocker cell (GRID_SIZE-1, GRID_SIZE-2) is hard-walled in
    ``_try_generate``; together with the locked door this isolates the
    exit pocket. Property: removing the door from the graph isolates the
    exit; if Hypothesis ever finds a back door, the level design is
    broken.
    """
    ws = generate_world(seed, agent_ids)
    start = next(iter(ws.agent_positions.values()))
    seen: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            n = (nx, ny)
            if n in seen:
                continue
            # Pretend door is impassable; locked-door semantics.
            if n == ws.door_position:
                continue
            if ws.grid[ny][nx] == CellType.EMPTY:
                seen.add(n)
                queue.append(n)
    assert ws.exit_position not in seen, (
        f"exit {ws.exit_position} reachable without unlocking the door (seed={seed})"
    )


@given(seed=_SEEDS, agent_ids=_AGENT_IDS)
@_PROPERTY_SETTINGS
def test_initial_invariants(seed: int, agent_ids: list[str]) -> None:
    """Inventories empty, door locked, no agent at exit, turn=0, status=running."""
    ws = generate_world(seed, agent_ids)
    assert ws.turn == 0
    assert ws.status == "running"
    assert ws.door_locked is True
    assert ws.key_holder is None
    assert ws.agents_at_exit == set()
    for aid in agent_ids:
        assert ws.agent_inventories[aid] == set(), f"{aid} starts non-empty"
        assert ws.stuck_counter[aid] == 0
        assert ws.inboxes[aid] == []
    # The classifier and stuck-detection logic both assume DEFAULT_TURN_LIMIT
    # is positive; if someone ever zeroes it the smoke test will hang.
    assert DEFAULT_TURN_LIMIT > 0
