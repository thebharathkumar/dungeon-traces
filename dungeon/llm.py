"""LLM call wrapper with tenacity-driven retry.

A single network blip during a Phase 4 run currently kills the run and
loses the spent tokens. This module wraps ``client.messages.create`` with
exponential backoff + jitter so transient failures (rate limits, 5xx,
connection resets) cost a retry, not the whole run.

Design notes:
    * Only **transient** errors are retried. ``BadRequestError`` and the
      auth/permission family are permanent — retrying them just burns
      tokens and obscures the real failure.
    * Tunables come from environment variables so a Phase 4 batch run can
      be re-run with longer waits without code changes.
    * Logs (not prints) so callers can suppress with ``--log-level``.

Environment variables (all optional):
    DUNGEON_RETRY_MAX              max attempts                (default 5)
    DUNGEON_RETRY_INITIAL_WAIT     initial wait, seconds        (default 1.0)
    DUNGEON_RETRY_MAX_WAIT         max wait between retries     (default 30.0)
    DUNGEON_RETRY_MULTIPLIER       exponential multiplier       (default 2.0)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)


def _retryable_exceptions() -> tuple[type[BaseException], ...]:
    """Anthropic exception classes that are worth retrying.

    Imported lazily so the module is importable when the SDK is missing
    (e.g. during offline test collection that never makes a real call).
    """
    try:
        from anthropic import (  # type: ignore[import-not-found]
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except ImportError:
        return ()
    return (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid %s=%r; falling back to %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid %s=%r; falling back to %s", name, raw, default)
        return default


def _log_attempt(retry_state: RetryCallState) -> None:
    """Tenacity ``before_sleep`` hook: emit one info line per retry."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    next_wait = retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
    logger.warning(
        "LLM call attempt %d failed (%s); retrying in %.2fs",
        retry_state.attempt_number,
        type(exc).__name__ if exc else "unknown",
        next_wait,
    )


def call_with_retry(client: Any, **kwargs: Any) -> Any:
    """Call ``client.messages.create(**kwargs)`` with exponential-backoff retry.

    Permanent errors (``BadRequestError``, auth, quota) propagate
    immediately; transient errors trigger a retry up to ``max_attempts``.
    Returns the SDK response object on success.
    """
    max_attempts = _env_int("DUNGEON_RETRY_MAX", 5)
    initial = _env_float("DUNGEON_RETRY_INITIAL_WAIT", 1.0)
    max_wait = _env_float("DUNGEON_RETRY_MAX_WAIT", 30.0)
    multiplier = _env_float("DUNGEON_RETRY_MULTIPLIER", 2.0)

    retry_types = _retryable_exceptions()
    if not retry_types:
        # Anthropic SDK isn't installed (offline test collection, etc.). The
        # call itself will fail loudly without retry, which is the right
        # behaviour: there is nothing recoverable to wait for.
        return client.messages.create(**kwargs)

    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(
            multiplier=initial,
            max=max_wait,
            exp_base=multiplier,
        ),
        retry=retry_if_exception_type(retry_types),
        before_sleep=_log_attempt,
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            return client.messages.create(**kwargs)
    # tenacity guarantees one of the attempts returns or reraises; this
    # line is unreachable but keeps mypy happy.
    raise RuntimeError("unreachable: tenacity exhausted without raising")
