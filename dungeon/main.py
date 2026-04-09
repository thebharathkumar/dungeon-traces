"""CLI entrypoint for running a single dungeon simulation.

Usage:
    python -m dungeon.main --seed 42
    python -m dungeon.main --seed 7 --turn-limit 30 --quiet
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .game import ConsoleLogger, run_game
from .world import render_ascii


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        from anthropic import Anthropic
    except ImportError:
        print("anthropic package is not installed. Run: pip install -r requirements.txt")
        return 2

    parser = argparse.ArgumentParser(description="Run a dungeon-agents simulation.")
    parser.add_argument("--seed", type=int, default=None, help="world generation seed")
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model id (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument("--turn-limit", type=int, default=60)
    parser.add_argument("--quiet", action="store_true", help="suppress per-turn output")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 2

    seed = args.seed if args.seed is not None else int(time.time())
    print(f"Starting run: seed={seed} model={args.model} turn_limit={args.turn_limit}")

    client = Anthropic()
    logger = ConsoleLogger(quiet=args.quiet)

    ws = run_game(
        seed=seed,
        client=client,
        model=args.model,
        turn_limit=args.turn_limit,
        logger=logger,
    )

    print()
    print("Final map:")
    print(render_ascii(ws))
    print()
    print(f"Final status: {ws.status}")
    print(f"Turns played: {ws.turn}")
    for aid, pos in ws.agent_positions.items():
        inv = sorted(ws.agent_inventories[aid])
        print(f"  agent {aid}: position={pos} inventory={inv}")
    return 0 if ws.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
