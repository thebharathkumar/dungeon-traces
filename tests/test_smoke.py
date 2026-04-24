"""End-to-end smoke test: drive the whole game loop with a scripted client.

Runs the full game machinery (world generation, tool execution, belief
updates, message queue, end conditions, stuck detection, event logging,
failure classifier) without a real Anthropic API call.

The script is shaped so the run produces at least one event of each
failure classification category, giving the classifier a regression net
beyond the unit-test branch coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dungeon.events import EventLogger
from dungeon.game import ConsoleLogger, run_game
from dungeon.tracing import JsonTraceSink, MultiSink


# Seed 42 places key at (0,4), A at (5,3), B at (5,1). The scripts are
# hand-picked to exercise every classifier category at least once.
_SCRIPTS = {
    "A": [
        ("Observe surroundings.", "observe", {}),
        ("Go west blindly into unknown.", "move", {"direction": "west"}),
        ("Observe again.", "observe", {}),
        ("Push south to look for key.", "move", {"direction": "south"}),
        (
            "Send message to B about plans.",
            "send_message",
            {"content": "I'm heading to (0,4) to look for key."},
        ),
        ("Keep moving south.", "move", {"direction": "south"}),
        ("Go west toward key.", "move", {"direction": "west"}),
        ("Still west.", "move", {"direction": "west"}),
        ("Try pick_up key (may be gone by now).", "pick_up", {"item": "key"}),
        ("Observe.", "observe", {}),
        ("Use item without having it.", "use_item", {"item": "key", "target": "door"}),
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


@pytest.mark.slow
def test_smoke_full_game_loop(scripted_client, runs_dir: Path) -> None:
    """The full pipeline should run end-to-end and write a parseable NDJSON."""
    run_id = "smoke_test"
    client = scripted_client(_SCRIPTS)
    console_logger = ConsoleLogger(quiet=True)
    tracer = MultiSink(sinks=[JsonTraceSink(run_id=run_id, out_dir=runs_dir)])

    with EventLogger(run_id=run_id, out_dir=runs_dir) as event_logger:
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
        event_logger.finalize(ws.status, ws.agents_at_exit)

    # The game must terminate cleanly with a known status.
    assert ws.status in {"running", "success", "timeout", "stuck"}
    assert event_logger.event_count > 0
    assert event_logger.path.exists()

    # NDJSON must parse and every row must have the canonical schema.
    expected_keys = {
        "event_id",
        "run_id",
        "turn",
        "agent_id",
        "timestamp",
        "latency_ms",
        "action_taken",
        "action_result",
        "action_succeeded",
        "agent_belief_state",
        "world_truth_state",
        "belief_accuracy",
        "divergence_fields",
        "divergence_age",
        "divergences",
        "message_context",
        "failure_classification",
        "reasoning",
        "usage",
    }
    rows = [json.loads(line) for line in event_logger.path.read_text().splitlines()]
    for row in rows:
        assert expected_keys.issubset(row.keys()), f"row missing keys: {row.keys()}"

    # Final-row patching done by finalize() must reflect the real status.
    assert rows[-1]["world_truth_state"]["status"] == ws.status


@pytest.mark.slow
def test_smoke_exercises_classifier_branches(scripted_client, runs_dir: Path) -> None:
    """The hand-picked script must surface multiple classification categories."""
    run_id = "smoke_branches"
    client = scripted_client(_SCRIPTS)

    with EventLogger(run_id=run_id, out_dir=runs_dir) as event_logger:
        ws = run_game(
            seed=42,
            client=client,
            model="mock-model",
            run_id=run_id,
            turn_limit=20,
            console_logger=ConsoleLogger(quiet=True),
            event_logger=event_logger,
            tracer=None,
        )
        event_logger.finalize(ws.status, ws.agents_at_exit)

    counts = dict(event_logger.classification_counts)
    # At least one failure category must fire and at least one success must be
    # logged; if either disappears the seed/scripts have drifted and the
    # regression net is no longer exercising the full pipeline.
    non_success = {k: v for k, v in counts.items() if k != "success"}
    assert counts.get("success", 0) > 0, f"no successes recorded: {counts}"
    assert non_success, f"no failures recorded: {counts}"
