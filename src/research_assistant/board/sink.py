"""Where board changes are mirrored. The graph state stays the source of truth."""

from collections.abc import Sequence
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
