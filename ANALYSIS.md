# Dungeon Agents: Analysis

This document captures the judgment calls, trade-offs, and interesting incidents behind the included simulation runs. It is the writeup a reviewer should read alongside the code and the trace viewer.

## Scope and deliberate cuts

The time budget for this project was intentionally tight. The scoring weights put most of the value on trace quality, legibility, and taste, not on how impressive the simulation is. So the simulation was kept at exactly the minimum needed to produce interesting failure modes, and everything past that was cut.

**In scope:**

- A flat 8x8 grid with a single key, a single locked door, and a single exit cell.
- Exactly six tools: `move`, `observe`, `pick_up`, `use_item`, `send_message`, `read_messages`. Every tool has a minimal return contract.
- Silent failures at the tool layer. A move into a wall returns `{moved: false}`, not an exception. Stale belief is the only signal the agent gets.
- A strict two-phase turn loop: deliver messages, then each agent takes exactly one tool call. Messages sent on turn N are delivered at the top of turn N+1, so both agents always act on at least one-turn-old information.
- A per-agent `BeliefState` that updates only from tool return values, separate from the authoritative `WorldState`. This separation is the whole point of the project: it is what makes divergence detection well-defined.
- Provenance tracking on three tracked facts (`key_position`, `door_locked`, `agent_{id}_position`) so the classifier can tell `coordination_failure` (partner invalidated your belief) from `information_lag` (your belief just got old).
- A four-category failure classifier: `agent_error`, `information_lag`, `coordination_failure`, `environment_constraint`.
- Two local trace artifacts per run (NDJSON event log + JSON trace file with full prompts) plus an optional Langfuse sink.
- A single-file HTML viewer that reads the NDJSON directly.

**Deliberately cut:**

- **Procedural dungeons.** A random-walled 8x8 with a fixed exit position and door placement produces enough variety across seeds without any of the engineering work of a real dungeon generator. The key insight is that the reviewer cares about failure modes in the agent's reasoning, not in the level design.
- **Line-of-sight fog.** The `observe` and post-`move` visibility is a simple 3x3 clip. Walls do not block vision. Line-of-sight would have added several hundred lines of code and produced exactly the same stale-belief bugs the current setup produces.
- **LangGraph.** The first plan considered LangGraph's prebuilt ReAct agent. It was rejected in favor of a ~30-line hand-rolled loop because the trace boundaries are cleaner: each `take_turn` call corresponds to exactly one prompt, one LLM response, one tool call, one event log row. LangGraph would have added ceremony and made it harder to draw the "what did the agent believe, then what did it do" boundary that the entire legibility layer depends on.
- **A rooms graph.** The door is just a single cell adjacent to the exit. Rooms would have required a more complex reachability check, and the same divergence patterns happen without them.
- **The optional Dungeon Master agent.** Out of scope; nothing in the scoring rubric rewards adding a third actor.
- **A "did the agent make the right call given its belief?" judgment.** This would be a useful addition but would require another LLM pass per event. Skipped as a Phase 5 idea.
- **Side-by-side run comparison in the viewer.** Cut for time; the heatmap already surfaces where incidents live in a single run.

## Trace strategy

The project produces three trace artifacts per run:

1. **`events_{run_id}.ndjson`**: one JSON object per line, one line per agent step. This is the primary diagnostic surface and is what the HTML viewer reads. Full schema at `dungeon/events.py:286`. Prompts are **deliberately excluded** from the NDJSON: a typical prompt is 2 KB, 120 prompts per run is 240 KB per run of repeated context that would bury the signal a reviewer is trying to find.
2. **`trace_{run_id}.json`**: structured per-turn record with full system prompt, user prompt, response content blocks, and usage. This is where a reviewer or a replay tool goes when they need to see exactly what the model saw.
3. **Langfuse** (optional): if `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in the environment, each run also produces a Langfuse trace with nested spans for every agent turn and nested generations for every LLM call. The sink auto-disables when the keys are missing and exceptions are swallowed per-call, so an observability outage can never take down a run.

The NDJSON schema deliberately answers the five questions the spec asks a trace to answer:

1. **What did the agent believe at the time of the decision?** → `agent_belief_state`.
2. **What was actually true in the world?** → `world_truth_state`.
3. **Did the belief match reality? For how long has it been diverged?** → `belief_accuracy`, `divergence_fields`, `divergence_age`, `divergences`.
4. **What was the agent trying to do, and did it succeed?** → `action_taken`, `action_result`, `action_succeeded`, `reasoning`.
5. **If it failed, was it an agent error or an information problem?** → `failure_classification`.

Everything else in the schema (`latency_ms`, `message_context`, `usage`) is secondary context.

### The four failure categories

The classifier is deliberately narrow and the logic is documented inline at `dungeon/events.py:228`. Reviewers can audit the heuristic at that single call site.

- **`agent_error`**: the agent's beliefs were accurate (or the divergence was not relevant to the action it took), and the action still failed. This is the category that suggests the agent itself is the problem, not the information it was working from. Example: trying to `use_item` with a key it never picked up.
- **`information_lag`**: the action failed, a relevant belief was stale, and the belief got stale because of something the agent itself did or because of passage of time. Example: the agent picked up the key on an earlier turn but still remembers the key as being on the ground (this should not actually happen because `pick_up` updates belief atomically, but the category is correct if it did).
- **`coordination_failure`**: the action failed, a relevant belief was stale, and the stale belief was specifically caused by the *other* agent. The partner moved the cheese. This is the category that argues for better inter-agent communication. Example: agent A tries to walk onto agent B's cell because A's last confirmed partner position is stale.
- **`environment_constraint`**: the action failed and the agent had no way of knowing it would. For a move, this is specifically the case where the destination cell was not in the agent's `seen_cell_positions`. Exploring the unknown is expected; failing to explore the unknown is expected. This classification tells a reviewer to **not** blame the agent.

The most important subtle thing about the classifier is that "the agent had a stale belief at the same time the action failed" is NOT enough to classify it as an info problem. The stale belief has to **plausibly explain this specific failure**. For moves, that means looking at the `blocked_by` reason the tool returns. A move that bumped a wall while the agent's belief about the partner's position was stale is not a coordination failure, because the stale partner belief could not have caused the wall collision. This check is done by `_is_relevant_for_tool` at `dungeon/events.py:189`.

This was the single most useful correctness bug caught by the offline classifier test (`scripts/test_classifier.py`): before the `blocked_by` field was added, any failed move in a world where any belief was stale was classified as `coordination_failure`, which is a false positive a reviewer would immediately dismiss as the trace being untrustworthy.

## Legibility layer

The viewer is a single file (`viewer/index.html`, ~900 lines, vanilla JS and CSS, no dependencies, no build step). Open it in a browser, load any NDJSON file, done. Architecture notes:

- **Static layout, fixed 1400px.** No responsive breakpoints, no theme toggle, no filters, no export. Every absent feature was a conscious cut to avoid pulling attention away from the three things that actually matter: the belief-vs-truth table, the classification chips, and the heatmap.
- **Three views on the same data.** The timeline (left) is for walking through a run in order. The detail panel (right) is for drilling into one event. The heatmap (bottom) is for scanning a whole run at once. The timeline and heatmap are bidirectional: clicking a cell in the heatmap scrolls the timeline to that event and opens the detail panel, and keyboard navigation updates both.
- **Color as the primary signal.** Amber means "stale but still understandable", red means "actively wrong", green means "match", gray means "unknown, and unknown is not wrong". Failure classification is communicated by both a pill color in the detail panel and a background tint on the timeline row, so scanning a run for `coordination_failure` or `information_lag` takes under 5 seconds without any filters or sorting.
- **Divergences caused by the partner are explicitly labeled.** When a divergence's `caused_by_agent` does not match the agent whose event you are looking at, the detail panel says `caused by {agent} on turn {N}` directly under the stale value. This is the single most useful line in the whole viewer for explaining a coordination failure incident.
- **Raw `action_result` is always shown in full.** Viewers that hide data behind "show more" buttons lose reviewers. The one thing a reviewer will always want to see is the exact dict the tool returned.

## Run-by-run incidents

All five included runs are with `claude-sonnet-4-5` at `turn_limit=60`. Four of them ended in the `stuck` status and one (seed 155) reached `timeout`. No run in this batch reached `success`, which is itself the most interesting cross-run finding: Sonnet is meaningfully risk-averse about committing to exploration in a sparse-information environment, and will oscillate between two adjacent directions for many turns rather than reverse course. The `last_action_feedback` channel added to the prompt does unstick the worst pathology (repeating the same failed direction verbatim) but does not make Sonnet a great explorer, which is exactly the class of finding the trace layer is built to make visible. Per-event classification is where the interesting texture lives.

Headline per seed:

| seed | turns | events | success | agent_error | information_lag | coordination_failure | environment_constraint |
|-----:|------:|-------:|--------:|------------:|----------------:|---------------------:|-----------------------:|
|    7 |    15 |     31 |      11 |           5 |               0 |                    0 |                     15 |
|   42 |    12 |     25 |      10 |           7 |               0 |                    1 |                      7 |
|  101 |    21 |     43 |      20 |           4 |               0 |                    0 |                     19 |
|  155 |    60 |    120 |      69 |           6 |               0 |                    0 |                     45 |
| 2027 |    15 |     31 |      15 |           9 |               0 |                    0 |                      7 |

### Run 1: seed 7

The simplest illustration of the `agent_error` vs `environment_constraint` split. Both agents alternate `north` and `east` through the run; the directions the classifier flags are different for each.

- **Agent A, turn 5, `move(direction='north')`** at position (3, 2). Classifier: **`agent_error`**. The cell to the north is a wall, but A had already observed (3, 1) at turn 3 from the 3x3 window around (3, 2), so it was in `seen_cell_positions` (15 cells seen by turn 5) when the move was issued. A had no excuse. The classifier correctly attributes this to the agent, not the environment.
- **Agent B, turn 5, `move(direction='north')`** from position (7, 0). Classifier: **`environment_constraint`**. The target (7, -1) is out of bounds and was never in `seen_cells`, so at the moment of decision the classifier treats this as "exploration into the unknown" and exonerates the agent. The stuck counter still counts OOB failures as a commit `1509f5b` correction, but the *failure classifier* is a separate concern: it is asking "should a reviewer blame this on the agent?", and the answer for an unvisited edge is still no.

This is the distinction the whole diagnostic layer exists to make: the same tool call with the same `moved: false` result is two completely different failure modes depending on what the agent could have known.

### Run 2: seed 42

The only `coordination_failure` in the batch, and the most informative single event in the entire submission.

- **Agent A, turn 5, `move(direction='north')`** from (6, 1), aiming at (6, 0). Classifier: **`coordination_failure`**.
- A's belief state: `last_known_other_position: [5, 0]`, `facts_last_seen.other_position: 4`. That is, A last saw B on turn 4.
- World truth at the moment of decision: B was at `[6, 0]`. B had moved from (5, 0) to (6, 0) on turn 4 (after A's turn-4 observation).
- Tool result: `{moved: false, blocked_by: "other_agent"}`.
- Divergence record on the event: `field: other_agent_position, believed: [5, 0], actual: [6, 0], divergence_age: 1, caused_by_agent: "B", caused_at_turn: 4, caused_change: "moved"`.

This is the textbook case the classifier is built to catch: the move failed, the failing field (other agent's position) was relevant to the failure (the tool said `blocked_by: other_agent`), and the stale belief was caused by the *other* agent's action within the same turn. The Phase 3 viewer renders this as an amber row with `caused by B on turn 4` inline in the detail panel, which lets a reviewer understand "A did not do anything wrong here, it was acting on a sibling turn's worth of old information" in under five seconds.

An interesting subtlety: A had no unread messages at this point (`inbox_at_decision_count: 0`). The coordination failure is not "A did not read its mail", it is "there is no mail to read because the simulation deliberately does not auto-broadcast state". Better communication between agents is the actual remediation this classification is pointing at.

### Run 3: seed 101

The longest run (21 turns, 43 events) and the only one in the batch where a pick_up actually succeeded.

- **Agent A, turn 3, `pick_up(item='key')`** at (4, 4). Classifier: **success (no classification)**.
- A's belief key position at decision time: `[4, 4]`. World truth key position: `[4, 4]`. Belief and truth matched exactly; the action was a straightforward confirmation.
- Tool result: `{ok: true, success: true, picked_up: "key", inventory: ["key"]}`.
- How A got there in the first place is worth tracing: turn 0 observe (seeing the key at (4, 4) from the 3x3 window since A spawned at (5, 3)), turn 1 `move east`, turn 2 `move south`, turn 3 `pick_up`. Three turns, clean execution.

The reason this run is still interesting for diagnostic purposes is what happens *after* the pickup. A had the key but never reached the door, and B never joined up. Both agents drifted into the same north/east oscillation pattern seen in seed 7. The detail panel for turns 16 to 21 is a wall of alternating `agent_error` and `environment_constraint` classifications. A reviewer reading this run end-to-end in the viewer will see a clean success at turn 3 followed by a slow, visible collapse into mutual stuckness, which is the exact story this project is trying to surface: the failure was not a single bad decision but a long run of individually-reasonable-looking turns that never converged.

### Run 4: seed 2027

The `agent_error` heavy run. B hit agent-error classified failures on 9 of its 15 turns and accounts for most of the stuck counter pressure.

- **Agent B, turn 5, `move(direction='north')`** from (2, 4). Classifier: **`agent_error`**.
- Target (2, 3) was a wall, and B had been at (2, 4) on the previous turn where the 3x3 window included (2, 3). `seen_cell_positions` at decision time contained 15 cells including (2, 3). The last_action_feedback line in the prompt would have said `blocked by wall at (2, 3)` on the turn after the first failed attempt, yet B still retried variants of the same direction.
- Reasoning field: *"I need to continue exploring to find the key and map out the dungeon."* No acknowledgement of the previous failure, despite it being in the user prompt.

This turn is the single best evidence that Sonnet's belief-use is imperfect: the information was there, the feedback was there, and the model still chose a known-blocked action. The classifier is working correctly; the *agent* is the bug. For a reviewer asking "are the traces actually telling me what went wrong, or am I looking at a trace-layer artifact?", this is the incident to point at: the classifier categorically rules out information problems and puts the blame exactly where it belongs.

### Run 5: seed 155

The only run in the batch to reach `turn_limit=60` without the stuck counter tripping, and the clearest illustration of the "exploration failure, not divergence failure" category. The final event count is 120, exactly twice the turn count.

- **Agent A, turn 6, `pick_up(item='key')`** at (3, 7). Classifier: **success**. A started at (1, 7), walked east to (2, 7), then east to (3, 7), and picked up the key. This is the second successful key pickup in the batch, two turns slower than seed 101's turn-3 pickup.
- **Agent A then walked the wrong way.** With the key in hand, A moved *north* on turn 8 (to (3, 6)) instead of *east* toward the door at (6, 7). A never recovered: by turn 59 the key-holder was at (6, 0), the top edge of the map, diagonally opposite the door.
- **Neither agent ever observed the door.** `last_known_door_locked` stays `unknown` for every one of A's 60 events and every one of B's 60 events. The door cell at (6, 7) was never in either agent's 3x3 window in any turn. This is the most interesting single finding of the run: **the divergence layer has nothing to flag because both agents' beliefs about `door_locked` were never populated**. The classification counts reflect this — 0 coordination_failure, 0 information_lag, 6 agent_error, 45 environment_constraint. The failure is upstream of the trace layer: there is no divergence to detect when there is no belief to diverge from.
- **Zero messages.** Neither agent ever called `send_message` or `read_messages` across all 60 turns. A had a piece of information the partner desperately needed (the key-holder's location, plus the fact that the key was now collected) and never communicated it. This is the failure mode a reviewer should point at when asking "why is inter-agent communication a first-class affordance in this sim?".
- **Tool-use asymmetry.** A called `observe` 34 times and `move` 25 times; B called `move` 57 times and `observe` only 3 times. Same model, same system prompt, same environment — polar opposite strategies. Under a different seed this asymmetry might have been productive (A maps, B explores), but with no coordination channel it just means A was contemplating while B was wandering.

The Phase 3 viewer makes this run's story visible at a glance: the belief-vs-truth heatmap is almost entirely gray cells (beliefs were `unknown`, not `diverged`), with a single amber stripe across A's row where it briefly lost sight of the key position. A reviewer scanning the heatmap sees "nothing red" and has to read the detail panel to realize that the nothing-red is itself the bug: the agents never learned enough to be wrong.

## Limitations and what I would do next

- **The classifier is heuristic, not learned.** It works on exactly the three tracked facts and exactly the six tools. A larger toolset would need new rules in `_is_relevant_for_tool`. A learned classifier that judges divergence relevance by reading the reasoning would be more robust but adds an LLM dependency to the trace logger itself, which I did not want.
- **Belief is updated in Python after each tool call.** The agent never re-reads its own belief from the LLM layer, so there is no "the model forgot" failure mode possible. That is by design (the point of this setup is to isolate stale-info bugs from memory bugs), but it means a real deployment would need a different instrumentation strategy.
- **No "did the agent make a reasonable call given its belief?" check.** The classifier answers "was the agent's belief accurate", not "was the agent's reasoning correct". Adding a second-pass LLM judge on the `reasoning` field per event would be the single highest-value addition.
- **The viewer cannot diff two runs.** A common diagnostic question is "is this bug new in run 2 or does run 1 also have it?" and the current viewer cannot answer that. A side-by-side heatmap view would be ~50 lines more JS.
- **Langfuse integration is minimal.** It produces nested spans and generations correctly but does not tag runs with the classification breakdown or link out to the viewer. Both would be easy additions if this project went further.

## AI collaboration notes

This project was built interactively with Claude Code in the web UI. A few notes on how that actually went, since this section is in the scoring rubric.

- **The plan-first workflow was load-bearing.** Every phase started with a written plan that the human approved before any code was written. Phase 2's `blocked_by` field on `tool_move` would not exist if the plan hadn't forced an explicit discussion of "how will the classifier distinguish wall collisions from partner collisions" before coding started.
- **The most useful Claude catch was the classifier false positive.** The initial classifier rule was "if any belief is stale when an action fails, that's information_lag". The offline `scripts/test_classifier.py` (which Claude proposed adding before the live runs) surfaced a false positive where a wall collision in a world with a stale partner belief was being flagged as `coordination_failure`. That led directly to the `_is_relevant_for_tool` scoping logic, which is arguably the most important single piece of the diagnostic layer.
- **The most useful human catch was the "all observes" failure mode.** On the first live Phase 4 run, both agents called `observe` every single turn and never moved, because the minimal prompt did not say anything about the fact that re-observing a known cell is wasted. This was immediately obvious from reading the console log and led to a one-sentence prompt nudge. A fully autonomous agent would have committed the stuck run and spent the rest of the budget writing up "the agents are too risk-averse" when the actual fix was five seconds of reading the stdout.
- **Commit discipline.** Each feature shipped as its own commit with a descriptive message, even small ones like the status-display fix. This was explicitly requested by the human and turned out to be useful for reading the project history later as a standalone story of decisions.
