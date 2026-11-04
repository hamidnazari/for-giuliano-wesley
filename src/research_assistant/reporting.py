"""Report helpers shared by every Writer: which cards are gaps, and a plain report layout."""

import re

from research_assistant.models import Board, Card, CardStatus

_MARKUP = re.compile(r"\[([^\]]*)\]\([^)]*\)|https?://\S+|[<>`#*_\[\]|]")


def _link_text(m: re.Match[str]) -> str:
    return m.group(1) if m.group(1) is not None else " "


def clean_title(title: str, limit: int = 120) -> str:
    """A page title made safe to print: no links, URLs, headings or line breaks.

    Titles come from web pages, so they are untrusted. Link text is kept; the target is not.
    """
    text = _MARKUP.sub(_link_text, title)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


_OPEN = (CardStatus.BACKLOG, CardStatus.IN_PROGRESS, CardStatus.IN_REVIEW)


_URL = re.compile(r"https?://\S+")


def _no_urls(text: str) -> str:
    """Gap text can carry URLs from error messages or quoted notes. Reports link only sources."""
    return _URL.sub("a web page", text)


def limitations(board: Board) -> list[str]:
    """Gaps the report must admit: failed or unfinished cards.

    A split parent is not a gap, because its children cover it.
    """
    parents = {c.parent_id for c in board.cards.values() if c.parent_id}
    gaps: list[str] = []
    for card in sorted(board.cards.values(), key=lambda c: c.id):
        if card.id in parents:
            continue
        if card.status in (CardStatus.BLOCKED, CardStatus.REJECTED):
            reason = _no_urls(card.reason)
            gaps.append(f"{card.id}: {card.sub_question} ({card.status.value}: {reason})")
        elif card.status in _OPEN:
            gaps.append(f"{card.id}: {card.sub_question} (not completed)")
        if card.status is CardStatus.DONE:
            gaps += [
                f"{card.id}: dropped after failing review: {_no_urls(line)}"
                for line in card.dropped
            ]
    return gaps


def done_cards(board: Board) -> list[Card]:
    return sorted(
        (c for c in board.cards.values() if c.status is CardStatus.DONE), key=lambda c: c.id
    )


def plain_report(board: Board, question: str, stop_reason: str | None) -> str:
    """A deterministic report from approved cards only. Used by the stub Writer."""
    lines = [f"# {question}", ""]
    if stop_reason:
        lines += [f"> This report is partial. The run stopped early: {stop_reason}", ""]
    cards = done_cards(board)
    for card in cards:
        lines += [f"## {card.sub_question}", ""]
        lines += [f"- {n.text} [{', '.join(n.source_ids)}]" for n in card.notes]
        lines.append("")
    lines += limitations_section(board)
    lines += sources_section(board)
    return "\n".join(lines).rstrip() + "\n"


def sources_section(board: Board) -> list[str]:
    """The Sources section for approved cards, or nothing when there are none."""
    entries = [
        f"- [{s.source_id}] {clean_title(s.title) or s.locator}: {s.locator}"
        if s.kind == "web"
        else f"- [{s.source_id}] {clean_title(s.title)} (your document, chunk {s.locator})"
        for card in done_cards(board)
        for s in card.sources
    ]
    return ["## Sources", "", *entries, ""] if entries else []


def limitations_section(board: Board, extra: list[str] | None = None) -> list[str]:
    gaps = [*limitations(board), *(extra or [])]
    return ["## Limitations", "", *[f"- {g}" for g in gaps], ""] if gaps else []
