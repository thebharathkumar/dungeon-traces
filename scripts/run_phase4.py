"""Phase 4 batch runner: a fixed set of seeds, one model, sequential.

This is the script a reviewer runs to reproduce the included run
artifacts. It is deliberately simple and not a general batch runner:
the seeds below were picked to surface an interesting mix of outcomes
(at least one success, at least one timeout or stuck, at least one
information_lag or coordination_failure classification). Changing the
seeds changes what the reviewer sees, so they are hard-coded.

Usage:
    python scripts/run_phase4.py

Artifacts land under runs/phase4/:
    events_seed{N}.ndjson   NDJSON event log (one line per agent step)
    summary_seed{N}.json    run-level summary (status, classification counts)
    trace_seed{N}.json      full prompt + response trace per turn
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dungeon.events import EventLogger  # noqa: E402
from dungeon.game import ConsoleLogger, run_game  # noqa: E402
from dungeon.tracing import build_default_tracer  # noqa: E402
from dungeon.world import render_ascii  # noqa: E402

SEEDS = [7, 42, 101, 2027]
# Haiku 4.5 was too cautious in Phase 4 pilots: it would call observe
# every turn and never commit to exploring, which starved the trace
# layer of the stale-belief failures it is built to diagnose. Sonnet 4.5
# is willing to move and to plan multiple turns ahead, which produces
# the coordination and information-lag failures we actually want to
# demonstrate. dungeon/main.py keeps haiku as its default for cheap
# single runs; this batch intentionally uses the stronger model.
MODEL = "claude-sonnet-4-5"
TURN_LIMIT = 60
OUT_DIR = Path("runs/phase4")


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

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = Anthropic()

    overall_t0 = time.perf_counter()
    results = []
    for seed in SEEDS:
        run_id = f"seed{seed}"
        print(f"\n{'=' * 60}\n=== run {run_id}  seed={seed}  model={MODEL}\n{'=' * 60}")

        console_logger = ConsoleLogger()
        event_logger = EventLogger(run_id=run_id, out_dir=OUT_DIR)
        tracer = build_default_tracer(run_id=run_id, out_dir=OUT_DIR)

        ws = None
        t0 = time.perf_counter()
        try:
            ws = run_game(
                seed=seed,
                client=client,
                model=MODEL,
                run_id=run_id,
                turn_limit=TURN_LIMIT,
                console_logger=console_logger,
                event_logger=event_logger,
                tracer=tracer,
            )
        finally:
            elapsed_s = time.perf_counter() - t0
            final_status = ws.status if ws is not None else "crashed"
            agents_at_exit = ws.agents_at_exit if ws is not None else None
            summary = {
                "run_id": run_id,
                "seed": seed,
                "model": MODEL,
                "turn_limit": TURN_LIMIT,
                "status": final_status,
                "turns_played": ws.turn if ws is not None else 0,
                "total_events": event_logger.event_count,
                "classification_counts": dict(event_logger.classification_counts),
                "wall_seconds": round(elapsed_s, 2),
                "events_path": str(event_logger.path),
            }
            event_logger.write_run_summary(summary)
            event_logger.finalize(final_status, agents_at_exit)

        if ws is not None:
            print()
            print("Final map:")
            print(render_ascii(ws))
            print()
            print(f"Final status: {ws.status}  turns played: {ws.turn}")
        print(f"Classifications: {dict(event_logger.classification_counts)}")
        results.append(summary)

    overall = time.perf_counter() - overall_t0
    print(f"\n{'=' * 60}")
    print(f"Phase 4 batch done in {overall:.1f}s")
    print(f"{'=' * 60}")
    for r in results:
        cls = r["classification_counts"]
        print(
            f"  seed {r['seed']:>4}: status={r['status']:<8} "
            f"turns={r['turns_played']:>2} events={r['total_events']:>3} "
            f"cls={cls}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
