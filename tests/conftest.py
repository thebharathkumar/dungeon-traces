"""Shared pytest fixtures.

The fixtures here exist for two reasons:

1. ``ScriptedClient`` lets us drive the entire game loop end-to-end without
   a real Anthropic API call. It is the offline test double for
   ``anthropic.Anthropic``: same surface (``client.messages.create(...)``),
   deterministic responses driven by a per-agent action queue.
2. ``belief_factory`` / ``truth_factory`` produce well-typed snapshot dicts
   for classifier unit tests so each test does not have to repeat the
   noisy boilerplate of building a belief / truth dict by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fake Anthropic client for offline end-to-end tests.
# ---------------------------------------------------------------------------


@dataclass
class FakeBlock:
    """One content block returned by a fake Anthropic response."""

    type: str
    text: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    id: str | None = None


@dataclass
class FakeUsage:
    input_tokens: int = 120
    output_tokens: int = 40


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    usage: FakeUsage = field(default_factory=FakeUsage)


@dataclass
class ScriptedClient:
    """Fake Anthropic client driven by a per-agent action queue.

    Each script entry is ``(reasoning, tool_name, tool_input)``. When the
    queue for an agent empties, the client falls back to ``observe`` so the
    game loop can keep advancing without crashing the test.
    """

    scripts: dict[str, list[tuple[str, str, dict[str, Any]]]]

    def __post_init__(self) -> None:
        self.scripts = {k: list(v) for k, v in self.scripts.items()}
        # Mirrors anthropic.Anthropic().messages.create(...)
        self.messages = self

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> FakeResponse:
        agent_id = "A" if "You are A" in system else "B"
        script = self.scripts.get(agent_id) or []
        if script:
            reasoning, tool_name, tool_input = script.pop(0)
        else:
            reasoning, tool_name, tool_input = ("no plan, just look", "observe", {})
        return FakeResponse(
            content=[
                FakeBlock(type="text", text=reasoning),
                FakeBlock(type="tool_use", name=tool_name, input=tool_input, id="mock"),
            ]
        )


@pytest.fixture
def scripted_client() -> Callable[[dict[str, list[tuple[str, str, dict[str, Any]]]]], ScriptedClient]:
    """Factory fixture: ``client = scripted_client({'A': [...], 'B': [...]})``."""

    def _make(scripts: dict[str, list[tuple[str, str, dict[str, Any]]]]) -> ScriptedClient:
        return ScriptedClient(scripts=scripts)

    return _make


# ---------------------------------------------------------------------------
# Snapshot factories for classifier unit tests.
# ---------------------------------------------------------------------------


@pytest.fixture
def belief_factory() -> Callable[..., dict[str, Any]]:
    """Build a flat belief snapshot with sensible defaults; override per test."""

    def _make(
        *,
        self_id: str = "A",
        position: tuple[int, int] = (3, 3),
        seen: list[tuple[int, int]] | None = None,
        key_pos: Any = "unknown",
        door_locked: Any = "unknown",
        other_pos: Any = "unknown",
        facts_last_seen: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        return {
            "self_id": self_id,
            "position": list(position),
            "inventory": [],
            "last_known_key_position": (
                key_pos
                if key_pos == "unknown"
                else list(key_pos) if isinstance(key_pos, tuple) else key_pos
            ),
            "last_known_door_locked": door_locked,
            "last_known_other_position": (
                other_pos if other_pos == "unknown" else list(other_pos)
            ),
            "facts_last_seen": facts_last_seen or {},
            "seen_cell_count": len(seen or []),
            "seen_cell_positions": sorted([list(p) for p in (seen or [])]),
        }

    return _make


@pytest.fixture
def truth_factory() -> Callable[..., dict[str, Any]]:
    """Build a flat ground-truth snapshot with sensible defaults; override per test."""

    def _make(
        *,
        key_position: tuple[int, int] | None = None,
        key_holder: str | None = None,
        door_locked: bool = True,
        agent_positions: dict[str, list[int]] | None = None,
    ) -> dict[str, Any]:
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

    return _make


# ---------------------------------------------------------------------------
# Filesystem fixture: isolate per-test run artifacts under tmp_path.
# ---------------------------------------------------------------------------


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    """Per-test temp directory in place of ``./runs``. Auto-cleaned by pytest."""
    d = tmp_path / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d
