"""Tests for ``dungeon.logging_setup.setup_logging``."""

from __future__ import annotations

import logging

from dungeon.logging_setup import setup_logging


def test_sets_root_level() -> None:
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    setup_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_unknown_level_falls_back_to_info() -> None:
    # caplog deliberately not used: caplog.at_level restores the root
    # level on exit, which would mask whether setup_logging applied the
    # fallback.
    setup_logging("BANANA")
    assert logging.getLogger().level == logging.INFO


def test_idempotent_does_not_stack_handlers() -> None:
    """Calling setup_logging twice must not double-emit log lines."""
    setup_logging("INFO")
    first = list(logging.getLogger().handlers)
    setup_logging("INFO")
    second = list(logging.getLogger().handlers)
    assert len(second) == 1, f"handlers stacked: was {len(first)}, now {len(second)}"
