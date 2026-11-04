"""Plain-text view of a board for the terminal."""

from research_assistant.models import Board, Card, CardStatus

_TITLES = {
    CardStatus.BACKLOG: "Backlog",
    CardStatus.IN_PROGRESS: "In progress",
    CardStatus.IN_REVIEW: "In review",
    CardStatus.DONE: "Done",
    CardStatus.BLOCKED: "Blocked",
    CardStatus.REJECTED: "Rejected",
}


def _line(card: Card, width: int) -> str:
    question = card.sub_question
    if len(question) > width:
        question = question[: width - 1] + "…"
    detail = f"  [{len(card.notes)} notes, {len(card.sources)} sources, attempts {card.attempts}]"
    text = f"  {card.id}  {question}{detail}"
    if card.reason:
        text += f"\n      reason: {card.reason}"
    return text


def render_board(board: Board, width: int = 60) -> str:
    lines = [f"Board {board.run_id}"]
    for status, title in _TITLES.items():
        cards = [c for c in board.cards.values() if c.status is status]
        lines.append(f"\n{title} ({len(cards)})")
        lines.extend(_line(c, width) for c in sorted(cards, key=lambda c: c.id))
    return "\n".join(lines)
