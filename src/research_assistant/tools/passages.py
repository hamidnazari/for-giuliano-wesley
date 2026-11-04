"""fetch_cited_passage and the mechanical checks the Critic runs on what it fetches.

The tool takes a source ID, never a URL. It looks the URL up in this run's source store, so
the Critic can only read pages the Researchers actually retrieved.
"""

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from research_assistant.guardrails.budget import Budget
from research_assistant.sources import SourceStore
from research_assistant.tools.search import ExtractBackend

MAX_PAGE_CHARS = 60_000


class UnknownSource(ValueError):
    pass


@dataclass(frozen=True)
class Passage:
    source_id: str
    kind: str
    locator: str
    text: str
    origin: str  # "page" (full text), "snippet" (search excerpt only) or "document"


class PassageCache:
    """One fetch per source per run, shared by every review round."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Passage] = {}

    def get(self, source_id: str) -> Passage | None:
        with self._lock:
            return self._items.get(source_id)

    def put(self, passage: Passage) -> None:
        with self._lock:
            self._items[passage.source_id] = passage


def make_fetch_cited_passage(
    sources: SourceStore, backend: ExtractBackend | None, cache: PassageCache, budget: Budget
) -> Callable[..., dict[str, Any]]:
    def fetch_cited_passage(source_id: str) -> dict[str, Any]:
        source = sources.get(source_id)
        if source is None:
            raise UnknownSource(source_id)
        passage = cache.get(source_id)
        if passage is None:
            if source.kind == "doc":  # already local, nothing to fetch
                passage = Passage(source_id, "doc", source.locator, source.snippet, "document")
            else:
                text = ""
                if backend is not None:
                    budget.before_fetch()  # BudgetExceeded must reach the graph, not be caught
                    text = backend.extract(source.locator)
                if text.strip():
                    passage = Passage(
                        source_id, "web", source.locator, text[:MAX_PAGE_CHARS], "page"
                    )
                else:  # the page could not be read; the search excerpt is all we have
                    passage = Passage(source_id, "web", source.locator, source.snippet, "snippet")
            cache.put(passage)
        return {
            "source_id": passage.source_id,
            "kind": passage.kind,
            "locator": passage.locator,
            "text": passage.text,
            "origin": passage.origin,
        }

    return fetch_cited_passage


# ---- mechanical checks ----

_QUOTE = re.compile(r'"([^"]{12,})"|“([^”]{12,})”')
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]{4,}")
_DASHES = str.maketrans({"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "\u00a0": " "})


def normalize(text: str) -> str:
    """Lowercase, one space between words, plain quotes and dashes, no markdown emphasis."""
    text = text.translate(_DASHES).lower().replace("*", "").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def missing_quotes(note: str, texts: list[str]) -> list[str]:
    """Quoted phrases in the note that appear in none of the cited texts."""
    haystack = normalize(" ".join(texts))
    quotes = [a or b for a, b in _QUOTE.findall(note)]
    return [q for q in quotes if normalize(q) not in haystack]


def missing_numbers(note: str, texts: list[str]) -> list[str]:
    """Numbers in the note that appear in none of the cited texts. A hint, not proof."""
    haystack = normalize(" ".join(texts)).replace(",", "")
    seen: list[str] = []
    for raw in _NUMBER.findall(note):
        num = raw.rstrip(",").replace(",", "")
        # whole numbers only: 5 must not match inside 50, 5.5 or 2.5
        found = re.search(rf"(?<![\d.]){re.escape(num)}(?!\d|\.\d)", haystack)
        if not found and num not in seen:
            seen.append(num)
    return seen


def best_windows(text: str, claim: str, count: int = 2, size: int = 1000) -> list[str]:
    """The stretches of a long text that share the most words with the claim."""
    if len(text) <= size:
        return [text] if text.strip() else []
    wanted = set(_WORD.findall(normalize(claim))) | {
        n.replace(",", "") for n in _NUMBER.findall(claim)
    }
    step = size // 2
    scored: list[tuple[int, int]] = []
    for start in range(0, len(text), step):
        window = normalize(text[start : start + size])
        words = set(_WORD.findall(window)) | {n.replace(",", "") for n in _NUMBER.findall(window)}
        scored.append((len(wanted & words), start))
    picked: list[int] = []
    for score, start in sorted(scored, key=lambda x: (-x[0], x[1])):
        if score == 0 and picked:
            break
        if all(abs(start - p) >= size for p in picked):
            picked.append(start)
        if len(picked) == count:
            break
    return [text[s : s + size].strip() for s in sorted(picked)]
