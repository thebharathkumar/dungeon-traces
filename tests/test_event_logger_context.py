"""Tests for ``EventLogger`` as a context manager.

The original API leaked the file handle if the surrounding code raised
before reaching ``finalize``. The context-manager wrapper guarantees the
file is closed even on exception, while still allowing ``finalize`` to be
called explicitly when the caller has the final world status to patch in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dungeon.events import EventLogger


def test_context_manager_closes_file_on_clean_exit(runs_dir: Path) -> None:
    with EventLogger(run_id="ctx-clean", out_dir=runs_dir) as logger:
        assert logger._fh is not None
        assert not logger._fh.closed
    assert logger._fh.closed


def test_context_manager_closes_file_on_exception(runs_dir: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with EventLogger(run_id="ctx-exc", out_dir=runs_dir) as logger:
            assert not logger._fh.closed
            raise RuntimeError("boom")
    assert logger._fh.closed


def test_finalize_inside_context_still_closes(runs_dir: Path) -> None:
    """If finalize is called inside the with block, __exit__ must be a no-op."""
    with EventLogger(run_id="ctx-finalize", out_dir=runs_dir) as logger:
        # finalize() rewrites the file and closes the handle. __exit__ then
        # sees an already-closed handle and must not raise.
        logger.finalize("success", agents_at_exit=set())
        assert logger._fh.closed
    assert logger._fh.closed
