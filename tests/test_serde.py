"""Tests for the shared ``jsonify`` helper.

Both agent.py and events.py used to ship their own copy of this; this
module is what the dedupe protects. The contract is small but worth
locking down so a future change in one call site does not surprise the
other.
"""

from __future__ import annotations

import json

from dungeon._serde import jsonify


def test_tuple_becomes_list() -> None:
    assert jsonify((1, 2)) == [1, 2]
    assert jsonify(()) == []


def test_passthrough_for_primitives() -> None:
    for v in (None, True, 0, 3.14, "hello"):
        assert jsonify(v) is v


def test_passthrough_for_lists_and_dicts() -> None:
    """jsonify intentionally does NOT recurse; nested tuples stay tuples.

    The codebase already speaks in JSON-native types at the boundary, so
    the only conversion that matters is top-level tuple->list (positions).
    Recursive coercion would be premature; if a future field needs it,
    add it here with a focused test.
    """
    nested = [(1, 2), (3, 4)]
    out = jsonify(nested)
    assert out is nested  # same object
    assert isinstance(out[0], tuple)


def test_output_is_json_serializable() -> None:
    pos = (5, 7)
    assert json.dumps(jsonify(pos)) == "[5, 7]"
