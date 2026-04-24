# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and dates use
ISO-8601.

## [Unreleased]

The `super-boost` branch — a hardening pass that adds proper packaging,
test infrastructure, observability, and CI without changing the agent or
classifier behaviour. Every commit is independently revertable.

### Added

- **`pyproject.toml`** with PEP 621 metadata, pinned `[project.optional-dependencies].dev` group (`pytest`, `pytest-cov`, `hypothesis`, `mypy`, `ruff`, `tenacity`), and tool config sections for ruff, mypy, pytest, and coverage. Existing `requirements.txt` is preserved for users who only want to run.
- **`requirements-dev.txt`** mirroring the dev extras for users who don't want an editable install.
- **`.pre-commit-config.yaml`** wiring ruff (`check --fix`) and `ruff-format` as pre-commit hooks.
- **`tests/`** directory with proper pytest layout:
  - `tests/conftest.py` — `runs_dir` and `scripted_client` fixtures.
  - `tests/test_classifier.py` — port of `scripts/test_classifier.py` (now table-driven and parametrized).
  - `tests/test_smoke.py` — full game-loop smoke against a scripted fake LLM.
  - `tests/test_world_properties.py` — Hypothesis property tests for world generation: determinism, reachability, no-walls, no-overlap, door-isolates-exit, initial invariants. 200 examples per property.
  - `tests/test_belief_consistency.py` — Hypothesis property tests for belief invariants under arbitrary action sequences (self-position never stale, seen_cells monotonic).
  - `tests/test_llm.py` — flaky-client tests for `call_with_retry`.
  - `tests/test_event_logger_context.py` — file-handle close on clean exit, on exception, after explicit finalize.
  - `tests/test_config.py` — `WorldConfig` immutability and plumbing.
  - `tests/test_logging_setup.py` — log level setting and handler idempotency.
  - `tests/test_serde.py` — locks down the shared `jsonify` contract.
- **`dungeon/llm.py`** — `call_with_retry()` wraps `client.messages.create()` with tenacity-driven exponential backoff + jitter. Retries `APITimeoutError`, `APIConnectionError`, `RateLimitError`, `InternalServerError`. Tunable via `DUNGEON_RETRY_*` env vars.
- **`dungeon/config.py`** — frozen `WorldConfig` dataclass + `DEFAULT_WORLD_CONFIG` singleton. Tunables: `wall_density`, `stuck_threshold`, `turn_limit`, `max_generation_retries`.
- **`dungeon/logging_setup.py`** — `setup_logging(level)` with idempotent handler registration.
- **`dungeon/_serde.py`** — canonical `jsonify` helper, replacing two duplicate inline copies.
- **`--log-level` CLI flag** in `dungeon.main` (default `WARNING`).
- **`EventLogger` context-manager protocol** — `__enter__` / `__exit__` guarantee the NDJSON file handle is closed on exception.
- **`.github/workflows/ci.yml`** — matrix on Python 3.10/3.11/3.12 running ruff, ruff-format check, mypy (best-effort), and pytest. Separate coverage job uploads `coverage.xml` as an artifact. Pip-cached and concurrency-cancelled.

### Changed

- `generate_world(seed, agent_ids)` → `generate_world(seed, agent_ids, config=DEFAULT_WORLD_CONFIG)`. Backward compatible — defaults reproduce the original behaviour.
- `run_game(...)` accepts an optional `config` kwarg; an explicit `turn_limit` still wins for backward compatibility with the CLI.
- `WorldState` now carries the `config` it was generated from, so `_check_end` reads `ws.config.stuck_threshold` instead of a module-level constant.
- `dungeon/agent.py`'s `take_turn` routes its `messages.create` call through `call_with_retry`.
- `dungeon/main.py` uses `with EventLogger(...)` so a crash mid-run no longer leaks the file handle.
- `dungeon/tracing.py` — the two `print()` calls in `_safe` and `MultiSink._fanout` that silently swallowed sink errors are now `logger.warning` calls so they show up under `--log-level WARNING`.
- `compute_divergences`, `belief_accuracy_map`, and `classify_failure` in `dungeon/events.py` now type-annotate `belief_snapshot` / `truth_snapshot` / `provenance` as `Mapping[str, Any]` (previously `dict`) so static type checkers flag any future mutation attempt.
- `scripts/smoke_test.py` and `scripts/test_classifier.py` now delegate to `pytest` instead of duplicating logic. Outputs are unchanged.
- `README.md` documents the new dev install (`pip install -e .[dev]`), the `--log-level` flag, the retry env vars, and the `WorldConfig` knobs.

### Removed

- Two inline copies of `_jsonify` (in `dungeon/agent.py` and `dungeon/events.py`) — replaced by the single source in `dungeon/_serde.py`.
- Unused `dataclasses.field` import in `dungeon/events.py` flagged by ruff F401.
