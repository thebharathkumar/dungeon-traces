"""Offline smoke test: run the game loop against a scripted fake LLM client.

This does not call the real Anthropic API. It exists so we can verify the
entire plumbing (world generation, tool execution, belief updates, message
queue, end conditions, stuck detection) without spending tokens and without
needing ANTHROPIC_API_KEY. The real entrypoint is dungeon/main.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dungeon.game import ConsoleLogger, run_game  # noqa: E402
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

    Each entry is (reasoning, tool_name, tool_input). When an agent runs out
    of scripted actions, it falls back to 'observe'.
    """

    def __init__(self, scripts: dict[str, list[tuple[str, str, dict]]]) -> None:
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.messages = self  # so client.messages.create works

    def create(self, *, model, max_tokens, system, tools, messages):
        # Figure out which agent this call is for by reading the system prompt.
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
    scripts = {
        "A": [
            ("I'll look around first.", "observe", {}),
            ("Try going south to explore.", "move", {"direction": "south"}),
            ("Tell B I'm exploring south.", "send_message", {"content": "Exploring south, will look for key."}),
            ("Keep moving south.", "move", {"direction": "south"}),
            ("Head west toward unknown.", "move", {"direction": "west"}),
            ("Read any messages now.", "read_messages", {}),
            ("Move west again.", "move", {"direction": "west"}),
            ("Pick up if there's a key.", "pick_up", {"item": "key"}),
            ("Head back east.", "move", {"direction": "east"}),
            ("Try using key on door.", "use_item", {"item": "key", "target": "door"}),
        ],
        "B": [
            ("I should observe.", "observe", {}),
            ("Go east.", "move", {"direction": "east"}),
            ("Check for messages.", "read_messages", {}),
            ("Move south.", "move", {"direction": "south"}),
            ("Keep going south.", "move", {"direction": "south"}),
            ("Try west.", "move", {"direction": "west"}),
        ],
    }

    client = ScriptedClient(scripts)
    logger = ConsoleLogger()
    ws = run_game(
        seed=42,
        client=client,
        model="mock-model",
        turn_limit=15,
        logger=logger,
    )
    print()
    print("Final map:")
    print(render_ascii(ws))
    print()
    print(f"Final status: {ws.status}")
    print(f"Turns played: {ws.turn}")
    for aid, pos in ws.agent_positions.items():
        print(f"  {aid}: pos={pos} inv={sorted(ws.agent_inventories[aid])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
