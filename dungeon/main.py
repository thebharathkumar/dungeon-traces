"""CLI entrypoint for running a single dungeon simulation.

Usage:
    python -m dungeon.main --seed 42
    python -m dungeon.main --seed 7 --turn-limit 30 --quiet

Artifacts produced per run, under ./runs/:
    events_{run_id}.ndjson   structured event log (one line per agent step)
    summary_{run_id}.json    run-level summary (outcome, counts, final state)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from .events import EventLogger
from .game import ConsoleLogger, run_game
from .tracing import build_default_tracer
from .world import render_ascii


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
    parser.add_argument("--out-dir", default="runs", help="output directory for event and summary files")
    parser.add_argument("--run-id", default=None, help="override the generated run id")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 2

    seed = args.seed if args.seed is not None else int(time.time())
    run_id = args.run_id or _make_run_id()
    print(f"Starting run {run_id}: seed={seed} model={args.model} turn_limit={args.turn_limit}")

    client = Anthropic()
    console_logger = ConsoleLogger(quiet=args.quiet)
    tracer = build_default_tracer(run_id=run_id, out_dir=args.out_dir)

    ws = None
    with EventLogger(run_id=run_id, out_dir=args.out_dir) as event_logger:
        try:
            ws = run_game(
                seed=seed,
                client=client,
                model=args.model,
                run_id=run_id,
                turn_limit=args.turn_limit,
                console_logger=console_logger,
                event_logger=event_logger,
                tracer=tracer,
            )
        finally:
            final_status = ws.status if ws is not None else "crashed"
            agents_at_exit = ws.agents_at_exit if ws is not None else None
            summary = {
                "run_id": run_id,
                "seed": seed,
                "model": args.model,
                "turn_limit": args.turn_limit,
                "status": final_status,
                "turns_played": event_logger.events_in_memory[-1]["turn"]
                if event_logger.events_in_memory
                else 0,
                "total_events": event_logger.event_count,
                "classification_counts": dict(event_logger.classification_counts),
                "events_path": str(event_logger.path),
            }
            summary_path = event_logger.write_run_summary(summary)
            event_logger.finalize(final_status, agents_at_exit)
            trace_path = None
            for sink in getattr(tracer, "sinks", []):
                if hasattr(sink, "path") and sink.path is not None:
                    trace_path = sink.path
                    break
            print(f"\nWrote events to {event_logger.path}")
            print(f"Wrote summary to {summary_path}")
            if trace_path:
                print(f"Wrote trace  to {trace_path}")

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
