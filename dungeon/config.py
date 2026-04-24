"""Run-time configuration for the dungeon world.

The original code embedded tunables (wall density, stuck threshold, turn
limit) as module-level constants in ``world.py``. That works for a
single setting but makes A/B sweeps painful: every variant has to monkey
patch a global, which is not safe across parallel runs.

``WorldConfig`` is a frozen dataclass so it can be shared safely between
threads and used as a dict key for caching. The default instance
(:data:`DEFAULT_WORLD_CONFIG`) reproduces the original behaviour exactly,
so existing callers are unaffected.

Note: ``GRID_SIZE`` is intentionally *not* configurable here. The level
design (door + blocker forming an exit pocket at the bottom-right
corner) hard-codes the 8x8 layout in multiple places; making grid size
truly configurable is a larger refactor for a future commit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    """Tunables for world generation and game-loop termination.

    Attributes:
        wall_density: Probability that any non-reserved cell is a wall.
            Higher values produce harder mazes; values above ~0.30 make
            connectivity hard to satisfy and inflate generation retries.
        stuck_threshold: Turns of zero-progress (no semantically successful
            tool call) after which an agent is flagged ``stuck``. The
            game ends when *all* agents are stuck.
        turn_limit: Hard cap on game turns; reaching it ends the game
            with status ``timeout``.
        max_generation_retries: How many times ``generate_world`` will
            re-roll the layout before giving up. 500 is generous; on
            wall_density=0.17 the first attempt succeeds ~95% of seeds.
    """

    wall_density: float = 0.17
    stuck_threshold: int = 6
    turn_limit: int = 60
    max_generation_retries: int = 500


DEFAULT_WORLD_CONFIG: WorldConfig = WorldConfig()
"""Singleton default config; identity-comparable so callers can detect
"caller passed a custom config" without an extra flag."""
