"""Unit tests for ``dungeon.llm.call_with_retry``.

These tests do not contact the real Anthropic API. They drive the retry
loop with a flaky fake client whose ``messages.create`` is rigged to fail
N times then succeed. The point is to lock down behaviour for:

* permanent errors (BadRequestError) propagate immediately
* transient errors (RateLimitError) are retried and eventually succeed
* exhausting retries raises the underlying exception, not a tenacity wrapper
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

# anthropic is required for the test even though the helper is import-safe
# without it; we want to actually exercise the retry path.
anthropic = pytest.importorskip("anthropic")

from dungeon.llm import call_with_retry  # noqa: E402


class _FlakyMessages:
    """Fake ``client.messages`` that raises ``error`` for the first ``fails`` calls."""

    def __init__(self, fails: int, error: Exception, success: Any = "ok") -> None:
        self.fails = fails
        self.error = error
        self.success = success
        self.calls = 0

    def create(self, **_kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self.fails:
            raise self.error
        return self.success


class _FlakyClient:
    def __init__(self, messages: _FlakyMessages) -> None:
        self.messages = messages


def _make_rate_limit_error() -> Exception:
    """Build a RateLimitError without depending on the SDK's __init__ shape.

    The Anthropic SDK's APIStatusError subclasses require a real
    httpx.Response to construct via the public API. For a unit test we
    only need an instance whose ``isinstance`` check passes, so we
    bypass __init__ via __new__.
    """
    err = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
    Exception.__init__(err, "rate limited")
    return err


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force tiny waits so retry tests finish in milliseconds."""
    monkeypatch.setenv("DUNGEON_RETRY_MAX", "5")
    monkeypatch.setenv("DUNGEON_RETRY_INITIAL_WAIT", "0.001")
    monkeypatch.setenv("DUNGEON_RETRY_MAX_WAIT", "0.005")
    monkeypatch.setenv("DUNGEON_RETRY_MULTIPLIER", "1.0")


def test_returns_immediately_on_first_success() -> None:
    msgs = _FlakyMessages(fails=0, error=RuntimeError("never raised"), success="hello")
    client = _FlakyClient(msgs)
    result = call_with_retry(client, model="m", max_tokens=1, system="", tools=[], messages=[])
    assert result == "hello"
    assert msgs.calls == 1


def test_retries_then_succeeds_on_transient_error() -> None:
    err = _make_rate_limit_error()
    msgs = _FlakyMessages(fails=2, error=err, success="recovered")
    client = _FlakyClient(msgs)
    result = call_with_retry(client, model="m", max_tokens=1, system="", tools=[], messages=[])
    assert result == "recovered"
    assert msgs.calls == 3, f"expected 3 calls (2 fail + 1 success), got {msgs.calls}"


def test_propagates_permanent_error_without_retry() -> None:
    """BadRequestError is not in the retryable set; it must surface immediately."""
    perm = anthropic.BadRequestError.__new__(anthropic.BadRequestError)
    Exception.__init__(perm, "bad request")
    msgs = _FlakyMessages(fails=10, error=perm, success="never")
    client = _FlakyClient(msgs)
    with pytest.raises(anthropic.BadRequestError):
        call_with_retry(client, model="m", max_tokens=1, system="", tools=[], messages=[])
    assert msgs.calls == 1, f"expected 1 call (no retry), got {msgs.calls}"


def test_exhausts_retries_and_reraises_original() -> None:
    err = _make_rate_limit_error()
    msgs = _FlakyMessages(fails=99, error=err, success="never")
    client = _FlakyClient(msgs)
    with pytest.raises(anthropic.RateLimitError):
        call_with_retry(client, model="m", max_tokens=1, system="", tools=[], messages=[])
    # DUNGEON_RETRY_MAX=5 so we get exactly 5 attempts.
    assert msgs.calls == 5, f"expected 5 attempts, got {msgs.calls}"


def test_no_anthropic_sdk_falls_through_without_retry() -> None:
    """When anthropic isn't importable, the helper still calls the client once.

    Simulates an offline import failure by monkey-patching the helper's
    exception lookup to return an empty tuple.
    """
    msgs = _FlakyMessages(fails=0, error=RuntimeError("unused"), success="bare-call")
    client = _FlakyClient(msgs)
    with patch("dungeon.llm._retryable_exceptions", return_value=()):
        result = call_with_retry(client, model="m", max_tokens=1, system="", tools=[], messages=[])
    assert result == "bare-call"
    assert msgs.calls == 1
