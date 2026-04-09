"""Offline smoke test: run the game loop against a scripted fake LLM client.

This does not call the real Anthropic API. It exists so we can verify the
entire plumbing (world generation, tool execution, belief updates, message
queue, end conditions, stuck detection, event logging, failure classifier)
without spending tokens and without needing ANTHROPIC_API_KEY.

The script is intentionally designed so the agents produce at least one
event in each failure classification category so the Phase 3 viewer and
the classifier itself can be eyeballed against real output.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dungeon.events import EventLogger  # noqa: E402
from dungeon.game import ConsoleLogger, run_game  # noqa: E402
from dungeon.tracing import JsonTraceSink, MultiSink  # noqa: E402
from dungeon.world import render_ascii  # noqa: E402


@dataclass
class _FakeBlock:
    type: str
    text: str | None = None
    name: str | None = None
    input: dict | None = None
    id: str | None = None


@dataclass
class _FakeUsage:
    input_tokens: int = 120
    output_tokens: int = 40


@dataclass
class _FakeResponse:
    content: list
    usage: _FakeUsage


class ScriptedClient:
    """Fake Anthropic client driven by a per-agent action queue.

    Each entry is (reasoning, tool_name, tool_input). When an agent runs
    out of scripted actions, it falls back to 'observe'.
    """

    def __init__(self, scripts: dict[str, list[tuple[str, str, dict]]]) -> None:
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.messages = self

    def create(self, *, model, max_tokens, system, tools, messages):
        agent_id = "A" if "You are A" in system else "B"
        script = self.scripts.get(agent_id) or []
        if script:
            reasoning, tool_name, tool_input = script.pop(0)
        else:
            reasoning, tool_name, tool_input = ("no plan, just look", "observe", {})
        content = [
            _FakeBlock(type="text", text=reasoning),
            _FakeBlock(type="tool_use", name=tool_name, input=tool_input, id="mock"),
        ]
        return _FakeResponse(content=content, usage=_FakeUsage())


def main() -> int:
    # Seed 42 places key at (0,4), A at (5,3), B at (5,1). The scripts below
    # are hand-picked to exercise each classifier category at least once.
    scripts = {
        "A": [
            ("Observe surroundings.", "observe", {}),
            ("Go west blindly into unknown.", "move", {"direction": "west"}),  # env constraint if wall
            ("Observe again.", "observe", {}),
            ("Push south to look for key.", "move", {"direction": "south"}),
            ("Send message to B about plans.", "send_message", {"content": "I'm heading to (0,4) to look for key."}),
            ("Keep moving south.", "move", {"direction": "south"}),
            ("Go west toward key.", "move", {"direction": "west"}),
            ("Still west.", "move", {"direction": "west"}),
            ("Try pick_up key (may be gone by now).", "pick_up", {"item": "key"}),
            ("Observe.", "observe", {}),
            ("Use item without having it.", "use_item", {"item": "key", "target": "door"}),  # agent_error
        ],
        "B": [
            ("Observe.", "observe", {}),
            ("Go south.", "move", {"direction": "south"}),
            ("Read messages.", "read_messages", {}),
            ("Head south.", "move", {"direction": "south"}),
            ("Go south.", "move", {"direction": "south"}),
            ("Go west.", "move", {"direction": "west"}),
            ("Go west more.", "move", {"direction": "west"}),
            ("Go west more.", "move", {"direction": "west"}),
            ("Observe for key.", "observe", {}),
            ("Try pick up key if here.", "pick_up", {"item": "key"}),
        ],
    }

    run_id = "smoke_test"
    out_dir = Path("runs")
    client = ScriptedClient(scripts)
    console_logger = ConsoleLogger()
    event_logger = EventLogger(run_id=run_id, out_dir=out_dir)
    tracer = MultiSink(sinks=[JsonTraceSink(run_id=run_id, out_dir=out_dir)])

    ws = run_game(
        seed=42,
        client=client,
        model="mock-model",
        run_id=run_id,
        turn_limit=20,
        console_logger=console_logger,
        event_logger=event_logger,
        tracer=tracer,
    )
    summary = {
        "run_id": run_id,
        "seed": 42,
        "status": ws.status,
        "turns_played": ws.turn,
        "total_events": event_logger.event_count,
        "classification_counts": dict(event_logger.classification_counts),
    }
    event_logger.write_run_summary(summary)
    event_logger.close()

    print()
    print("Final map:")
    print(render_ascii(ws))
    print()
    print(f"Final status: {ws.status}")
    print(f"Turns played: {ws.turn}")
    print(f"Events written: {event_logger.event_count}")
    print(f"Classification counts: {dict(event_logger.classification_counts)}")
    print(f"NDJSON: {event_logger.path}")

    # Spot-check the first and last events for schema completeness.
    with open(event_logger.path) as fh:
        lines = fh.readlines()
    print(f"\nFirst event keys: {sorted(json.loads(lines[0]).keys())}")
    print(f"Last event classification: {json.loads(lines[-1]).get('failure_classification')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
