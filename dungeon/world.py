"""Ground-truth world state for the dungeon.

The world knows everything. Agents only learn about it through tool calls
that return locally-scoped observations. The split between this module
and BeliefState in agent.py is what makes the belief-vs-truth divergence
observable in Phase 2 traces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

GRID_SIZE = 8
WALL_DENSITY = 0.17
STUCK_THRESHOLD = 2
DEFAULT_TURN_LIMIT = 60


class CellType(str, Enum):
    EMPTY = "empty"
    WALL = "wall"


Pos = tuple[int, int]


@dataclass
class Message:
    sender: str
    recipient: str
    content: str
    sent_turn: int
    deliver_turn: int


@dataclass
class WorldState:
    grid: list[list[CellType]]
    key_position: Optional[Pos]
    door_position: Pos
    exit_position: Pos
    door_locked: bool
    agent_positions: dict[str, Pos]
    agent_inventories: dict[str, set[str]]
    key_holder: Optional[str]
    inboxes: dict[str, list[Message]]
    pending_messages: list[Message]
    agents_at_exit: set[str] = field(default_factory=set)
    turn: int = 0
    status: str = "running"
    stuck_counter: dict[str, int] = field(default_factory=dict)

    def in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def cell_type(self, pos: Pos) -> CellType:
        x, y = pos
        return self.grid[y][x]

    def is_passable(self, pos: Pos) -> bool:
        if not self.in_bounds(pos):
            return False
        if self.cell_type(pos) == CellType.WALL:
            return False
        if pos == self.door_position and self.door_locked:
            return False
        return True

    def agent_at(self, pos: Pos) -> Optional[str]:
        for aid, p in self.agent_positions.items():
            if p == pos:
                return aid
        return None

    def describe_cell(self, pos: Pos) -> dict:
        """Dict describing what is visibly present at a cell. Used by tools."""
        x, y = pos
        contents = []
        if self.key_position == pos:
            contents.append("key")
        if self.door_position == pos:
            contents.append("door_locked" if self.door_locked else "door_open")
        if self.exit_position == pos:
            contents.append("exit")
        other = self.agent_at(pos)
        if other is not None:
            contents.append(f"agent:{other}")
        return {
            "x": x,
            "y": y,
            "type": self.cell_type(pos).value,
            "contents": contents,
        }

    def visible_cells(self, pos: Pos) -> list[dict]:
        """3x3 fog of war centered on pos, clipped at grid edges."""
        x, y = pos
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                p = (x + dx, y + dy)
                if self.in_bounds(p):
                    out.append(self.describe_cell(p))
        return out

    def ground_truth_snapshot(self) -> dict:
        """Flat snapshot of world state for event logging in Phase 2."""
        return {
            "turn": self.turn,
            "agent_positions": {aid: list(p) for aid, p in self.agent_positions.items()},
            "agent_inventories": {aid: sorted(inv) for aid, inv in self.agent_inventories.items()},
            "key_position": list(self.key_position) if self.key_position else None,
            "key_holder": self.key_holder,
            "door_position": list(self.door_position),
            "door_locked": self.door_locked,
            "exit_position": list(self.exit_position),
            "agents_at_exit": sorted(self.agents_at_exit),
            "status": self.status,
        }


def generate_world(seed: int, agent_ids: list[str]) -> WorldState:
    """Generate a playable world. Retries until connectivity constraints are met."""
    rng = random.Random(seed)
    for _ in range(500):
        ws = _try_generate(rng, agent_ids)
        if ws is not None:
            return ws
    raise RuntimeError(f"Failed to generate a valid world from seed {seed}")


def _chebyshev(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _try_generate(rng: random.Random, agent_ids: list[str]) -> Optional[WorldState]:
    grid = [[CellType.EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    exit_pos: Pos = (GRID_SIZE - 1, GRID_SIZE - 1)
    door_pos: Pos = (GRID_SIZE - 2, GRID_SIZE - 1)
    blocker: Pos = (GRID_SIZE - 1, GRID_SIZE - 2)
    reserved = {exit_pos, door_pos, blocker}

    # Wall off the second approach to the exit so the door is the only way in.
    grid[blocker[1]][blocker[0]] = CellType.WALL

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if (x, y) in reserved:
                continue
            if rng.random() < WALL_DENSITY:
                grid[y][x] = CellType.WALL

    empties = [
        (x, y)
        for y in range(GRID_SIZE)
        for x in range(GRID_SIZE)
        if grid[y][x] == CellType.EMPTY and (x, y) not in (exit_pos, door_pos)
    ]
    rng.shuffle(empties)
    if len(empties) < len(agent_ids) + 2:
        return None

    # Key: anywhere far enough from the exit pocket to force actual exploration.
    key_pos: Optional[Pos] = None
    for cand in empties:
        if _manhattan(cand, exit_pos) >= 4:
            key_pos = cand
            break
    if key_pos is None:
        return None
    empties.remove(key_pos)

    agent_positions: dict[str, Pos] = {}
    for aid in agent_ids:
        placed = False
        for cand in list(empties):
            if cand == key_pos:
                continue
            if _chebyshev(cand, exit_pos) < 4:
                continue
            if any(_chebyshev(cand, p) <= 1 for p in agent_positions.values()):
                continue
            agent_positions[aid] = cand
            empties.remove(cand)
            placed = True
            break
        if not placed:
            return None

    if not _connected(grid, key_pos, door_pos, exit_pos, agent_positions):
        return None

    return WorldState(
        grid=grid,
        key_position=key_pos,
        door_position=door_pos,
        exit_position=exit_pos,
        door_locked=True,
        agent_positions=agent_positions,
        agent_inventories={aid: set() for aid in agent_ids},
        key_holder=None,
        inboxes={aid: [] for aid in agent_ids},
        pending_messages=[],
        stuck_counter={aid: 0 for aid in agent_ids},
    )


def _connected(
    grid: list[list[CellType]],
    key_pos: Pos,
    door_pos: Pos,
    exit_pos: Pos,
    agent_positions: dict[str, Pos],
) -> bool:
    """BFS treating door as passable; verify all critical points are reachable."""
    start = next(iter(agent_positions.values()))
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            n = (nx, ny)
            if n in seen:
                continue
            if grid[ny][nx] == CellType.EMPTY or n == door_pos:
                seen.add(n)
                stack.append(n)
    required = {key_pos, door_pos, exit_pos} | set(agent_positions.values())
    return required.issubset(seen)


def render_ascii(ws: WorldState) -> str:
    """Human-readable rendering of the full world for CLI debugging."""
    out_lines = []
    for y in range(GRID_SIZE):
        row = []
        for x in range(GRID_SIZE):
            pos = (x, y)
            aid = ws.agent_at(pos)
            if aid is not None:
                row.append(aid[0])
                continue
            if ws.cell_type(pos) == CellType.WALL:
                row.append("#")
                continue
            if pos == ws.exit_position:
                row.append("X")
                continue
            if pos == ws.door_position:
                row.append("D" if ws.door_locked else "d")
                continue
            if pos == ws.key_position:
                row.append("k")
                continue
            row.append(".")
        out_lines.append(" ".join(row))
    return "\n".join(out_lines)
