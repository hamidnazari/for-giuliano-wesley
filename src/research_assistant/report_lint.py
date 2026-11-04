"""Structural checks run before a report is written: citations, links and empty sections."""

import re
from dataclasses import dataclass

from research_assistant.models import Board
from research_assistant.reporting import done_cards

CITE = re.compile(r"\[([A-Za-z0-9][\w.\-]*)\](?!\()")
URL = re.compile(r"https?://[^\s)>\]]+")
EXEMPT_SECTIONS = {"limitations", "sources"}


@dataclass(frozen=True)
class Claim:
    line: int  # 1-based line number in the report
    text: str
    cited: tuple[str, ...]


def known_sources(board: Board) -> tuple[set[str], set[str]]:
    """Source IDs and web URLs that approved cards actually have."""
    ids: set[str] = set()
    urls: set[str] = set()
    for card in done_cards(board):
        for s in card.sources:
            ids.add(s.source_id)
            if s.kind == "web":
                urls.add(s.locator)
    return ids, urls


def _heading(line: str) -> tuple[int, str] | None:
    m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
    return (len(m.group(1)), m.group(2)) if m else None


def claims(markdown: str) -> list[Claim]:
    """Body lines that assert something: not headings, quotes, or Limitations and Sources."""
    found: list[Claim] = []
    section: str | None = None
    for n, line in enumerate(markdown.splitlines(), start=1):
        head = _heading(line)
        if head:
            level, text = head
            section = text.lower() if level == 2 else (None if level == 1 else section)
            continue
        text = line.strip()
        if not text or text.startswith(">") or text in {"---", "***"}:
            continue
        if section in EXEMPT_SECTIONS:
            continue
        found.append(Claim(n, text, tuple(CITE.findall(text))))
    return found


def lint_report(markdown: str, board: Board) -> list[str]:
    """Return a readable list of problems. Empty means the report passes."""
    ids, urls = known_sources(board)
    issues: list[str] = []
    for c in claims(markdown):
        preview = c.text[:60]
        if not c.cited:
            issues.append(f"line {c.line}: uncited claim: {preview!r}")
        for cid in c.cited:
            if cid not in ids:
                issues.append(f"line {c.line}: unknown source [{cid}]")
    for n, line in enumerate(markdown.splitlines(), start=1):
        for found in URL.findall(line):
            url = found.rstrip(".,;:!?")
            if url not in urls:
                issues.append(f"line {n}: link not from an approved source: {url}")
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        head = _heading(line)
        if not head:
            continue
        nxt = next(
            ((j, _heading(x)) for j, x in enumerate(lines[i + 1 :], i + 1) if x.strip()), None
        )
        if nxt is None or (nxt[1] is not None and nxt[1][0] <= head[0]):
            issues.append(f"line {i + 1}: empty section {head[1]!r}")
    return issues
