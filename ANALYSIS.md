# Dungeon Agents: Analysis

This document captures the judgment calls, trade-offs, and interesting incidents behind the included simulation runs. It is the writeup a reviewer should read alongside the code and the trace viewer.

> Note: the run-by-run sections are stubbed while the live runs execute. They will be filled in once `runs/phase4/` contains real artifacts.

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

> These sections are written after the live runs in `runs/phase4/` complete. Each section picks one turn per run that best illustrates what the trace layer is surfacing and walks through it: the state, the action, the reasoning, and why the classifier labeled it the way it did.

### Run 1: seed 7

_pending live run data_

### Run 2: seed 42

_pending live run data_

### Run 3: seed 101

_pending live run data_

### Run 4: seed 2027

_pending live run data_

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
