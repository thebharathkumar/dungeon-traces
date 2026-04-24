"""Tiny serialization helpers shared across the package.

Both ``agent.py`` and ``events.py`` previously defined their own
``_jsonify`` with identical bodies. Two copies of the same one-liner is
exactly the kind of thing that quietly drifts apart: somebody adds
``set`` handling here, forgets to do it there, and the NDJSON for one
field starts looking different from the trace JSON. Centralising it
removes that footgun.
"""

from __future__ import annotations

from typing import Any


def jsonify(value: Any) -> Any:
    """Coerce a few common Python types into JSON-friendly equivalents.

    Currently the only transformation is ``tuple -> list`` because the
    rest of the codebase already speaks in JSON-native types (str, int,
    float, bool, None, list, dict). New conversions go here so both the
    NDJSON event log and the trace JSON pick them up automatically.
    """
    if isinstance(value, tuple):
        return list(value)
    return value
