"""Centralised logging configuration.

Two callers need to control verbosity:

* The CLI (``dungeon.main``), via the ``--log-level`` flag.
* Library users importing ``dungeon`` from a notebook or another script.

The function below is idempotent: calling it twice replaces the previous
handlers instead of stacking them, so a notebook user can re-run a cell
without duplicate log lines. Format is intentionally compact — full
JSON structured logging would be nicer for production but adds a
dependency for what is currently a single-process research tool.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit ``level`` and above to stderr.

    Args:
        level: Standard logging level name (DEBUG, INFO, WARNING, ERROR,
            CRITICAL). Case-insensitive. Unknown values fall back to
            INFO with a warning.
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        logging.warning("unknown log level %r; defaulting to INFO", level)
        numeric = logging.INFO

    root = logging.getLogger()
    # Drop existing handlers so repeat calls don't duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(numeric)
