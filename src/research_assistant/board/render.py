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


def card_markdown(card: Card) -> str:
    """Readable markdown for one card, used as the description of its Kanzo task."""
    parts = [f"**Sub-question:** {card.sub_question}", f"**Status:** {card.status.value}"]
    if card.parent_id:
        parts.append(f"**Split from:** {card.parent_id}")
    if card.reason:
        parts.append(f"**Reason:** {card.reason}")
    if card.notes:
        parts.append(
            "**Notes**\n" + "\n".join(f"- {n.text} [{', '.join(n.source_ids)}]" for n in card.notes)
        )
    if card.sources:
        parts.append(
            "**Sources**\n"
            + "\n".join(
                f"- [{s.source_id}] {s.title or s.locator} ({s.locator})"
                + (
                    f", credibility {card.credibility[s.source_id]:.2f}"
                    if s.source_id in card.credibility
                    else ""
                )
                for s in card.sources
            )
        )
    if card.dropped:
        parts.append(
            "**Dropped after failing review**\n" + "\n".join(f"- {d}" for d in card.dropped)
        )
    if card.critic_feedback:
        parts.append("**Critic feedback**\n" + "\n".join(f"- {f}" for f in card.critic_feedback))
    return "\n\n".join(parts)
