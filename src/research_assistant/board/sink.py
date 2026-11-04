"""Where board changes are mirrored. The graph state stays the source of truth."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from research_assistant.board.ops import Op
from research_assistant.models import Board


class BoardSink(Protocol):
    def sync(self, board: Board, ops: Sequence[Op]) -> None:
        """Mirror ops that were just applied. board is the state after applying them."""
        ...


class InMemorySink:
    """Records everything it is given. Used in tests and as the default sink."""

    def __init__(self) -> None:
        self.ops: list[Op] = []
        self.latest: Board | None = None

    def sync(self, board: Board, ops: Sequence[Op]) -> None:
        self.ops.extend(ops)
        self.latest = board.model_copy(deep=True)


class JsonSnapshotSink:
    """Writes the latest board to board.json in the project folder, for the board command."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def sync(self, board: Board, ops: Sequence[Op]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(board.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic: a reader never sees a half-written file


def load_snapshot(path: Path) -> Board:
    return Board.model_validate_json(path.read_text(encoding="utf-8"))
