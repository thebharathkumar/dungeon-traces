"""Trace sinks: Langfuse + local JSON, wired through a MultiSink fanout.

Two reasons this module exists on top of the NDJSON event log:

1. Langfuse is the "observability tool" called out in the spec. When
   LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set, every run produces
   a trace with nested spans for each agent turn and nested generations
   for each LLM call.

2. Even without Langfuse, we still want a portable trace artifact that
   captures the full prompts and raw response content blocks so a
   reviewer can reconstruct exactly what the model saw. The NDJSON log
   deliberately omits prompts (one 2 KB prompt per turn buries the
   signal); the JSON trace is where that verbose detail lives.

Both sinks degrade to no-ops if their dependencies or env vars are
missing. Runs should succeed in any environment.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


class TraceSink(Protocol):
    def start_run(self, *, run_id: str, seed: int, model: str, metadata: dict) -> None: ...
    def start_turn(self, *, turn: int, agent_id: str) -> None: ...
    def log_llm_call(
        self,
        *,
        system: str,
        user_prompt: str,
        output_blocks: list,
        usage: dict | None,
        latency_ms: int | None,
        model: str,
    ) -> None: ...
    def log_tool_call(
        self,
        *,
        name: str,
        tool_input: dict,
        output: dict,
        latency_ms: int | None,
    ) -> None: ...
    def end_turn(self, *, outcome: dict) -> None: ...
    def end_run(self, *, status: str, summary: dict) -> None: ...


@dataclass
class JsonTraceSink:
    """In-memory trace builder that writes one JSON file per run on end_run.

    The schema matches what the Phase 4 submission needs: a single file
    that, combined with the NDJSON event log, fully reconstructs a run.
    """

    run_id: str
    out_dir: Path = field(default_factory=lambda: Path("runs"))
    _data: dict = field(default_factory=dict)
    _current_turn: dict | None = field(default=None)
    path: Path | None = field(default=None)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"trace_{self.run_id}.json"
        self._data = {
            "run_id": self.run_id,
            "schema_version": "1",
            "started_at": _now(),
            "turns": [],
        }

    def start_run(self, *, run_id: str, seed: int, model: str, metadata: dict) -> None:
        self._data["run_id"] = run_id
        self._data["seed"] = seed
        self._data["model"] = model
        self._data["metadata"] = metadata or {}

    def start_turn(self, *, turn: int, agent_id: str) -> None:
        self._current_turn = {
            "turn": turn,
            "agent_id": agent_id,
            "started_at": _now(),
            "llm": None,
            "tool": None,
            "outcome": None,
        }

    def log_llm_call(
        self,
        *,
        system: str,
        user_prompt: str,
        output_blocks: list,
        usage: dict | None,
        latency_ms: int | None,
        model: str,
    ) -> None:
        if self._current_turn is None:
            return
        self._current_turn["llm"] = {
            "model": model,
            "system": system,
            "user": user_prompt,
            "output_blocks": output_blocks,
            "usage": usage,
            "latency_ms": latency_ms,
        }

    def log_tool_call(
        self,
        *,
        name: str,
        tool_input: dict,
        output: dict,
        latency_ms: int | None,
    ) -> None:
        if self._current_turn is None:
            return
        self._current_turn["tool"] = {
            "name": name,
            "input": tool_input,
            "output": output,
            "latency_ms": latency_ms,
        }

    def end_turn(self, *, outcome: dict) -> None:
        if self._current_turn is None:
            return
        self._current_turn["outcome"] = outcome
        self._current_turn["ended_at"] = _now()
        self._data["turns"].append(self._current_turn)
        self._current_turn = None

    def end_run(self, *, status: str, summary: dict) -> None:
        self._data["status"] = status
        self._data["summary"] = summary or {}
        self._data["ended_at"] = _now()
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, default=_json_default)


class LangfuseSink:
    """Optional Langfuse observability sink.

    Tries to import langfuse and instantiate a client. If the import
    fails or LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are missing, every
    method becomes a no-op. Exceptions from the Langfuse SDK are caught
    so an observability outage can never take down a run.
    """

    def __init__(self) -> None:
        self._lf = None
        self._trace = None
        self._turn_span = None
        self._model: str | None = None
        try:
            from langfuse import Langfuse  # type: ignore

            if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
                self._lf = Langfuse()
        except Exception:
            self._lf = None

    @property
    def enabled(self) -> bool:
        return self._lf is not None

    def _safe(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # Langfuse failures are non-fatal: the run should still
            # complete with a local NDJSON trace. Surface at WARNING so
            # `--log-level WARNING` users see persistent breakage.
            logger.warning("langfuse sink error: %s", e)
            return None

    def start_run(self, *, run_id: str, seed: int, model: str, metadata: dict) -> None:
        if not self.enabled:
            return
        self._model = model
        self._trace = self._safe(
            self._lf.trace,
            name="dungeon_run",
            id=run_id,
            metadata={"seed": seed, "model": model, **(metadata or {})},
            tags=["dungeon-agents"],
        )

    def start_turn(self, *, turn: int, agent_id: str) -> None:
        if not self.enabled or self._trace is None:
            return
        self._turn_span = self._safe(
            self._trace.span,
            name=f"turn_{turn:02d}_{agent_id}",
            input={"turn": turn, "agent_id": agent_id},
        )

    def log_llm_call(
        self,
        *,
        system: str,
        user_prompt: str,
        output_blocks: list,
        usage: dict | None,
        latency_ms: int | None,
        model: str,
    ) -> None:
        if not self.enabled or self._turn_span is None:
            return
        self._safe(
            self._turn_span.generation,
            name="llm_call",
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            output=output_blocks,
            usage_details=usage or {},
            metadata={"latency_ms": latency_ms},
        )

    def log_tool_call(
        self,
        *,
        name: str,
        tool_input: dict,
        output: dict,
        latency_ms: int | None,
    ) -> None:
        if not self.enabled or self._turn_span is None:
            return
        self._safe(
            self._turn_span.span,
            name=f"tool_{name}",
            input=tool_input,
            output=output,
            metadata={"latency_ms": latency_ms},
        )

    def end_turn(self, *, outcome: dict) -> None:
        if not self.enabled or self._turn_span is None:
            return
        self._safe(self._turn_span.end, output=outcome)
        self._turn_span = None

    def end_run(self, *, status: str, summary: dict) -> None:
        if not self.enabled:
            return
        if self._trace is not None:
            self._safe(self._trace.update, output={"status": status, **(summary or {})})
        self._safe(self._lf.flush)


@dataclass
class MultiSink:
    """Fanout. Any exception raised by one sink is swallowed per-call."""

    sinks: list

    def _fanout(self, method: str, **kwargs) -> None:
        for sink in self.sinks:
            fn = getattr(sink, method, None)
            if fn is None:
                continue
            try:
                fn(**kwargs)
            except Exception as e:
                logger.warning("tracing sink %s.%s failed: %s", type(sink).__name__, method, e)

    def start_run(self, **kwargs) -> None:
        self._fanout("start_run", **kwargs)

    def start_turn(self, **kwargs) -> None:
        self._fanout("start_turn", **kwargs)

    def log_llm_call(self, **kwargs) -> None:
        self._fanout("log_llm_call", **kwargs)

    def log_tool_call(self, **kwargs) -> None:
        self._fanout("log_tool_call", **kwargs)

    def end_turn(self, **kwargs) -> None:
        self._fanout("end_turn", **kwargs)

    def end_run(self, **kwargs) -> None:
        self._fanout("end_run", **kwargs)


def build_default_tracer(run_id: str, out_dir: str | os.PathLike = "runs") -> MultiSink:
    """Always includes JsonTraceSink; adds LangfuseSink if its keys are set."""
    sinks: list = [JsonTraceSink(run_id=run_id, out_dir=Path(out_dir))]
    langfuse = LangfuseSink()
    if langfuse.enabled:
        sinks.append(langfuse)
    return MultiSink(sinks=sinks)
