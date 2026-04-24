"""Tests for ``WorldConfig`` plumbing through generate_world and run_game.

These don't aim for statistical correctness on wall density (that's
covered by Hypothesis property tests). The point is to verify the
config object is actually consulted, not silently ignored.
"""

from __future__ import annotations

import pytest

from dungeon.config import DEFAULT_WORLD_CONFIG, WorldConfig
from dungeon.world import generate_world


def test_default_config_matches_module_constants() -> None:
    """The dataclass defaults must equal the legacy module constants.

    If somebody tweaks one without the other, two callers see different
    "defaults" depending on which path they took.
    """
    from dungeon import world

    assert DEFAULT_WORLD_CONFIG.wall_density == world.WALL_DENSITY
    assert DEFAULT_WORLD_CONFIG.stuck_threshold == world.STUCK_THRESHOLD
    assert DEFAULT_WORLD_CONFIG.turn_limit == world.DEFAULT_TURN_LIMIT


def test_world_config_is_frozen() -> None:
    cfg = WorldConfig()
    with pytest.raises(Exception):  # FrozenInstanceError, but importable name varies
        cfg.wall_density = 0.5  # type: ignore[misc]


def test_generated_world_carries_config() -> None:
    """The config used at generation must be discoverable on the WorldState."""
    custom = WorldConfig(wall_density=0.05, stuck_threshold=99, turn_limit=200)
    ws = generate_world(seed=42, agent_ids=["A", "B"], config=custom)
    assert ws.config is custom
    assert ws.config.stuck_threshold == 99
    assert ws.config.turn_limit == 200


def test_generation_fails_loudly_on_unsatisfiable_config() -> None:
    """A 1.0 wall density makes layout impossible; the error must point to it."""
    impossible = WorldConfig(
        wall_density=1.0, max_generation_retries=3
    )
    with pytest.raises(RuntimeError, match=r"wall_density=1\.0"):
        generate_world(seed=1, agent_ids=["A", "B"], config=impossible)


def test_low_wall_density_succeeds_first_try_for_most_seeds() -> None:
    """Smoke check that wall_density actually affects the maze."""
    sparse = WorldConfig(wall_density=0.0, max_generation_retries=1)
    # Zero walls plus the single hard-coded blocker means generation
    # must succeed every time.
    for seed in range(20):
        ws = generate_world(seed=seed, agent_ids=["A", "B"], config=sparse)
        wall_cells = sum(
            1 for row in ws.grid for cell in row if cell.value == "wall"
        )
        # Only the bottom-right blocker is a wall.
        assert wall_cells == 1, f"seed={seed} unexpectedly had {wall_cells} walls"
