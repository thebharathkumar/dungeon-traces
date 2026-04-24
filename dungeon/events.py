"""Structured event log, divergence detection, and failure classifier.

This module is the diagnostic backbone of the project. Every agent step
produces exactly one Event that answers the five questions from the spec:

  1. What did the agent believe at the time of the decision?
  2. What was actually true in the world?
  3. Did the belief match reality? For how long has it been diverged?
  4. What was the agent trying to do, and did it succeed?
  5. If it failed, was it an agent error or an information problem?

Events are written as NDJSON (one JSON object per line). The Phase 3
viewer/incident report reads that file directly; neither reads trace JSON
or the Langfuse cloud, both of which exist for a different purpose
(replay / latency debugging / full prompts).

Intentional exclusions from NDJSON rows:
  - Full LLM prompt text. It lives in the trace JSON only. This keeps
    NDJSON scannable and prevents one 2KB prompt per turn from burying
    the signal.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .tools import DIRECTION_DELTAS, is_semantic_success

FAILURE_CATEGORIES = (
    "agent_error",
    "information_lag",
    "coordination_failure",
    "environment_constraint",
)

# Fields we track belief for. Everything else the agent knows is either
# always-accurate (own position, own inventory) or not worth diffing.
TRACKED_FACTS = ("key_position", "door_locked", "other_agent_position")


def _jsonify(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _other_agent_id(self_id: str, truth: dict) -> Optional[str]:
    for aid in truth.get("agent_positions", {}):
        if aid != self_id:
            return aid
    return None


def _actual_key_state(self_id: str, truth: dict) -> Any:
    """Resolve the canonical value of 'where is the key?' from ground truth.

    Matches the vocabulary an agent would use in its BeliefState so the
    equality check in compute_divergences is apples-to-apples.
    """
    holder = truth.get("key_holder")
    if holder == self_id:
        return "in_my_inventory"
    if holder is not None:
        return "in_partner_inventory"
    kp = truth.get("key_position")
    if kp is None:
        return "unknown"
    return list(kp)


def _key_matches(believed: Any, actual: Any) -> bool:
    if believed == actual:
        return True
    if isinstance(believed, list) and isinstance(actual, list):
        return list(believed) == list(actual)
    return False


def compute_divergences(
    belief_snapshot: dict,
    truth_snapshot: dict,
    provenance: dict,
    now_turn: int,
) -> list[dict]:
    """Diff a belief snapshot against ground truth. One entry per diverged fact.

    Facts the agent has never observed (value == 'unknown') are NOT reported
    as divergences. Unknown is not wrong, it is just unknown.
    """
    self_id = belief_snapshot["self_id"]
    facts_last_seen: dict = belief_snapshot.get("facts_last_seen", {})
    out: list[dict] = []

    # key_position
    believed_key = belief_snapshot.get("last_known_key_position")
    if believed_key != "unknown":
        actual_key = _actual_key_state(self_id, truth_snapshot)
        if not _key_matches(believed_key, actual_key):
            last_seen = facts_last_seen.get("key_position")
            prov = provenance.get("key_position", {})
            out.append(
                {
                    "field": "key_position",
                    "believed": _jsonify(believed_key),
                    "actual": _jsonify(actual_key),
                    "last_confirmed_turn": last_seen,
                    "divergence_age": (now_turn - last_seen) if last_seen is not None else None,
                    "caused_by_agent": prov.get("agent"),
                    "caused_at_turn": prov.get("turn"),
                    "caused_change": prov.get("change"),
                }
            )

    # door_locked
    believed_door = belief_snapshot.get("last_known_door_locked")
    if believed_door != "unknown":
        actual_door = truth_snapshot.get("door_locked")
        if believed_door != actual_door:
            last_seen = facts_last_seen.get("door_locked")
            prov = provenance.get("door_locked", {})
            out.append(
                {
                    "field": "door_locked",
                    "believed": believed_door,
                    "actual": actual_door,
                    "last_confirmed_turn": last_seen,
                    "divergence_age": (now_turn - last_seen) if last_seen is not None else None,
                    "caused_by_agent": prov.get("agent"),
                    "caused_at_turn": prov.get("turn"),
                    "caused_change": prov.get("change"),
                }
            )

    # other_agent_position
    believed_other = belief_snapshot.get("last_known_other_position")
    if believed_other != "unknown" and believed_other is not None:
        other = _other_agent_id(self_id, truth_snapshot)
        if other is not None:
            actual_other = list(truth_snapshot["agent_positions"][other])
            if list(believed_other) != actual_other:
                last_seen = facts_last_seen.get("other_position")
                prov = provenance.get(f"agent_{other}_position", {})
                out.append(
                    {
                        "field": "other_agent_position",
                        "believed": _jsonify(believed_other),
                        "actual": actual_other,
                        "last_confirmed_turn": last_seen,
                        "divergence_age": (now_turn - last_seen) if last_seen is not None else None,
                        "caused_by_agent": prov.get("agent"),
                        "caused_at_turn": prov.get("turn"),
                        "caused_change": prov.get("change"),
                    }
                )

    return out


def belief_accuracy_map(
    belief_snapshot: dict,
    divergences: list[dict],
) -> dict:
    """Per-fact label: 'match' | 'diverged' | 'unknown'."""
    diverged_fields = {d["field"] for d in divergences}
    out: dict[str, str] = {}
    for fact in TRACKED_FACTS:
        key = {
            "key_position": "last_known_key_position",
            "door_locked": "last_known_door_locked",
            "other_agent_position": "last_known_other_position",
        }[fact]
        value = belief_snapshot.get(key)
        if value == "unknown" or value is None:
            out[fact] = "unknown"
        elif fact in diverged_fields:
            out[fact] = "diverged"
        else:
            out[fact] = "match"
    return out


def _is_relevant_for_tool(
    tool_name: str,
    tool_input: dict,
    divergence: dict,
    result: dict,
) -> bool:
    """Which divergences plausibly CAUSED the failure of this action.

    Narrow by design: "the agent had a stale belief at the same time the
    action failed" is not enough; the stale belief has to plausibly explain
    THIS failure. For moves that means we look at the blocked_by reason
    returned by tool_move.
    """
    field_ = divergence["field"]
    if tool_name == "pick_up":
        return field_ == "key_position"
    if tool_name == "use_item":
        if tool_input.get("item") == "key" and field_ == "door_locked":
            return True
        if field_ == "key_position":
            return True
        return False
    if tool_name == "move":
        blocked = result.get("blocked_by")
        if blocked == "other_agent":
            return field_ == "other_agent_position"
        if blocked == "locked_door":
            return field_ == "door_locked"
        # Walls and edges are environmental; stale beliefs about partner
        # or key don't explain them.
        return False
    return False


def _destination_from(pos: list, direction: str) -> tuple[int, int]:
    dx, dy = DIRECTION_DELTAS.get(direction, (0, 0))
    return (pos[0] + dx, pos[1] + dy)


def classify_failure(
    *,
    tool_name: str,
    tool_input: dict,
    result: dict,
    belief_snapshot: dict,
    divergences: list[dict],
) -> Optional[str]:
    """Assign one of four categories, or None on success.

    Heuristic, documented here so reviewers can audit the reasoning:

      1. If the tool semantically succeeded, return None.
      2. Collect divergences that are RELEVANT to this tool+input
         (see _is_relevant_for_tool).
      3. If any relevant divergence was caused by a DIFFERENT agent
         than the one acting now, classify as 'coordination_failure'.
         Rationale: the partner moved the cheese.
      4. If any relevant divergence was present at all (caused by
         self, or provenance unknown), classify as 'information_lag'.
         Rationale: the agent was working from data that had gone
         out of date, but not because the partner invalidated it.
      5. For a failed move, if the destination cell was not in the
         agent's seen_cell_positions, classify as 'environment_constraint'.
         Rationale: the agent had no way of knowing the wall/door/edge
         was there; exploring into the unknown is expected.
      6. Otherwise the agent had accurate beliefs and the action still
         failed, so classify as 'agent_error'.
    """
    if is_semantic_success(tool_name, result):
        return None

    self_id = belief_snapshot["self_id"]
    relevant = [
        d
        for d in divergences
        if _is_relevant_for_tool(tool_name, tool_input, d, result)
    ]

    for d in relevant:
        caused_by = d.get("caused_by_agent")
        if caused_by and caused_by != self_id:
            return "coordination_failure"
    if relevant:
        return "information_lag"

    if tool_name == "move":
        pos = belief_snapshot.get("position", [0, 0])
        direction = tool_input.get("direction", "")
        dest = list(_destination_from(pos, direction))
        seen = belief_snapshot.get("seen_cell_positions", [])
        if dest not in seen:
            return "environment_constraint"

    return "agent_error"


@dataclass
class Event:
    event_id: str
    run_id: str
    turn: int
    agent_id: str
    timestamp: str
    latency_ms: dict
    action_taken: dict
    action_result: dict
    action_succeeded: bool
    agent_belief_state: dict
    world_truth_state: dict
    belief_accuracy: dict
    divergence_fields: list
    divergence_age: dict
    divergences: list
    message_context: dict
    failure_classification: Optional[str]
    reasoning: str
    usage: Optional[dict]


class EventLogger:
    """NDJSON writer, one file per run.

    Opens the file eagerly so a crashed run still leaves a partial log
    on disk. Also keeps a small in-memory summary so main.py can print
    it after the game ends without re-reading the NDJSON.

    Usable as a context manager so the file handle is guaranteed to be
    closed even if the surrounding code raises before reaching
    :meth:`finalize`::

        with EventLogger(run_id, out_dir) as logger:
            run_game(..., event_logger=logger)
            logger.finalize(ws.status, ws.agents_at_exit)
    """

    def __init__(self, run_id: str, out_dir: str | os.PathLike = "runs") -> None:
        self.run_id = run_id
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"events_{run_id}.ndjson"
        self._fh = open(self.path, "w", encoding="utf-8")
        self.event_count = 0
        self.classification_counts: Counter = Counter()
        self.events_in_memory: list[dict] = []

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> None:
        # Always close the file handle. ``finalize`` is *not* called
        # automatically because it needs the final world status, which
        # the caller owns; if the caller forgot, we still avoid a leak.
        self.close()

    def log_step(
        self,
        *,
        agent_id: str,
        turn: int,
        belief_snapshot: dict,
        truth_snapshot: dict,
        provenance: dict,
        record: dict,
        received_messages: list,
        sent_messages: list,
        unread_inbox_count: int,
    ) -> Event:
        divergences = compute_divergences(
            belief_snapshot=belief_snapshot,
            truth_snapshot=truth_snapshot,
            provenance=provenance,
            now_turn=turn,
        )
        accuracy = belief_accuracy_map(belief_snapshot, divergences)
        classification = classify_failure(
            tool_name=record.get("tool_name") or "",
            tool_input=record.get("tool_input") or {},
            result=record.get("tool_result") or {},
            belief_snapshot=belief_snapshot,
            divergences=divergences,
        )
        self.classification_counts[classification or "success"] += 1

        event = Event(
            event_id=str(uuid.uuid4()),
            run_id=self.run_id,
            turn=turn,
            agent_id=agent_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms={
                "llm": record.get("llm_latency_ms"),
                "tool": record.get("tool_latency_ms"),
                "total": (record.get("llm_latency_ms") or 0) + (record.get("tool_latency_ms") or 0),
            },
            action_taken={
                "tool_name": record.get("tool_name"),
                "tool_input": record.get("tool_input") or {},
            },
            action_result=record.get("tool_result") or {},
            action_succeeded=bool(record.get("semantic_success")),
            agent_belief_state=belief_snapshot,
            world_truth_state=truth_snapshot,
            belief_accuracy=accuracy,
            divergence_fields=[d["field"] for d in divergences],
            divergence_age={d["field"]: d["divergence_age"] for d in divergences},
            divergences=divergences,
            message_context={
                "inbox_at_decision_count": len(received_messages),
                "received_preview": received_messages[:5],
                "sent_this_turn": sent_messages,
                "unread_after_turn": unread_inbox_count,
            },
            failure_classification=classification,
            reasoning=record.get("reasoning") or "",
            usage=record.get("usage"),
        )
        self._write(event)
        return event

    def _write(self, event: Event) -> None:
        payload = asdict(event)
        self._fh.write(json.dumps(payload, default=_json_default))
        self._fh.write("\n")
        self._fh.flush()
        self.event_count += 1
        self.events_in_memory.append(payload)

    def write_run_summary(self, summary: dict) -> Path:
        """Write a run-level summary alongside the events file."""
        summary_path = self.out_dir / f"summary_{self.run_id}.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=_json_default)
        return summary_path

    def finalize(self, final_status: str, agents_at_exit: Optional[set] = None) -> None:
        """Patch the last event's world status and rewrite the NDJSON.

        Every row captures world_truth_state BEFORE its action, so even the
        last row stores status='running'. The viewer (and anyone reading the
        NDJSON standalone) needs to see the real final status without loading
        a second file, so once the run is over we patch the last event in
        memory and rewrite the file from events_in_memory.

        This also fixes agents_at_exit on the last row for success runs, for
        the same reason.
        """
        if self._fh and not self._fh.closed:
            self._fh.close()
        if not self.events_in_memory:
            return
        last = self.events_in_memory[-1]
        truth = last.setdefault("world_truth_state", {})
        truth["status"] = final_status
        if agents_at_exit is not None:
            truth["agents_at_exit"] = sorted(agents_at_exit)
        with open(self.path, "w", encoding="utf-8") as fh:
            for payload in self.events_in_memory:
                fh.write(json.dumps(payload, default=_json_default))
                fh.write("\n")

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)
