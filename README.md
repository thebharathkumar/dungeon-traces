# Dungeon Agents

Two LLM agents explore an 8x8 grid dungeon and try to cooperate to reach an exit. One agent has to find a key, unlock the door, and both have to step onto the exit cell together. Each agent only sees a 3x3 window around itself, and messages between them are delivered one turn late, so both agents are constantly acting on stale information.

The simulation itself is deliberately minimal. The point of this project is the **traces and the legibility layer** on top of those traces: per-agent belief state captured at every decision, a diff against ground truth, a four-category failure classifier, and a single-file HTML viewer that lets a reviewer scan a run for interesting incidents in seconds.

## Architecture

```
dungeon/world.py      8x8 grid, entity placement, ground-truth WorldState,
                      per-field provenance tracking, BFS reachability check
dungeon/tools.py      The six agent tools: move, observe, pick_up, use_item,
                      send_message, read_messages. Silent move failures, no
                      exceptions at the tool layer.
dungeon/agent.py      DungeonAgent + BeliefState. Belief updates only from
                      tool return values, never from ground truth.
dungeon/game.py       Turn scheduler, delayed message delivery, end-condition
                      checks, pre-action snapshots for the event logger.
dungeon/events.py     Divergence detection, four-category failure classifier,
                      NDJSON event log with one row per agent step.
dungeon/tracing.py    JSON trace sink (always on) and Langfuse sink (optional,
                      auto-disables without keys). Fanout through MultiSink.
dungeon/main.py       CLI entrypoint for single-run execution.
scripts/run_phase4.py Batch runner for the four included run artifacts.
scripts/smoke_test.py Offline smoke test against a scripted fake LLM client.
scripts/test_classifier.py  Unit-style checks for the failure classifier.
viewer/index.html     Single-file HTML/CSS/JS trace viewer. No dependencies,
                      no build step, opens NDJSON via a file picker.
```

## Install

Python 3.10+ is required (the code uses `str | os.PathLike` unions).

```
pip install -r requirements.txt
cp .env.example .env
# edit .env and fill in ANTHROPIC_API_KEY
```

The `langfuse` package is listed in `requirements.txt` but is optional at runtime. If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are not set, the Langfuse sink quietly disables itself and runs still produce the local NDJSON and JSON trace files.

## Run

Single run with your own seed:

```
python -m dungeon.main --seed 42
```

Optional flags: `--model <id>`, `--turn-limit <n>`, `--quiet`, `--out-dir <path>`, `--run-id <str>`.

Reproduce the four included Phase 4 runs:

```
python scripts/run_phase4.py
```

Offline smoke test (no API key needed, scripted fake client):

```
python scripts/smoke_test.py
```

Classifier unit tests:

```
python scripts/test_classifier.py
```

## View traces

Open `viewer/index.html` directly in a browser. No server, no build step.

Click **Load NDJSON** and pick any `runs/phase4/events_*.ndjson` file.

The viewer shows:

1. **Run summary bar.** Run id, final status, turn count, event count, start timestamp, and a classification breakdown: how many agent steps were each of `success`, `agent_error`, `information_lag`, `coordination_failure`, `environment_constraint`.
2. **Turn timeline (left).** One row per agent step. Row background tint encodes the failure classification. The dot on the right is green when the agent's belief matched reality at decision time, red when at least one tracked fact was diverged, gray when the agent had not yet observed any tracked facts. Click a row or use arrow keys (or j/k) to navigate.
3. **Detail panel (right).** For the selected step, shows the agent and turn metadata, the action taken and classification as a pill, the reasoning the LLM produced, and the belief-vs-truth table with three rows for `key_position`, `door_locked`, and `other_agent_position`. Diverged rows get an amber left border and `(stale N turns)` suffix; if the divergence was caused by the *other* agent (e.g. partner picked up the key since you last saw it), the row also shows `caused by {agent} on turn {N}`. The raw `action_result` JSON is displayed in full at the bottom. Message traffic is shown only when there is any, to keep the panel short.
4. **Belief divergence heatmap (bottom).** Full-width grid with two rows, one per agent. Each cell is a turn. Cell color is amber for 1 diverged fact, red for 2 or more; cells with no divergences but a failed action get a classification-tinted color. Click any cell to jump the timeline and detail panel to that event. This is the surface for scanning a run end-to-end in under a minute.

## Included runs

The four runs under `runs/phase4/` are the reference dataset for this submission. Model `claude-sonnet-4-5`, turn limit 60.

| seed | final status | turns | events | headline incident |
|-----:|:-------------|------:|-------:|:------------------|
|    7 | stuck        |    15 |     31 | `agent_error` vs `environment_constraint` side-by-side on the same move tool |
|   42 | stuck        |    12 |     25 | one `coordination_failure`: A walks into the cell B just moved to |
|  101 | stuck        |    21 |     43 | the only successful `pick_up` in the batch (A collects the key on turn 3) |
| 2027 | stuck        |    15 |     31 | 9 `agent_error` events on B demonstrating that even with last-action feedback, Sonnet sometimes retries known-blocked moves |

Per-turn incident writeups are in [ANALYSIS.md](./ANALYSIS.md#run-by-run-incidents).

## Where the judgment calls live

Architecture trade-offs, what is deliberately not built, the trace strategy, the legibility layer rationale, and the run-by-run incident writeups are all in [ANALYSIS.md](./ANALYSIS.md).
