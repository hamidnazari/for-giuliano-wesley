"""Per-run store of retrieved sources. The Critic later checks citations against it."""

import threading
from collections import defaultdict

from research_assistant.models import Source
from research_assistant.tools.search import SearchResult

_MAX_SNIPPET = 4000


class SourceStore:
    """Thread-safe: parallel Researchers add to it at the same time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, Source] = {}
        self._by_card_url: dict[tuple[str, str], str] = {}
        self._counts: defaultdict[str, int] = defaultdict(int)

    def add(self, card_id: str, result: SearchResult) -> Source:
        """Store a search result for a card. The same URL on the same card keeps its ID."""
        with self._lock:
            known = self._by_card_url.get((card_id, result.url))
            if known:
                return self._by_id[known]
            self._counts[card_id] += 1
            source = Source(
                source_id=f"{card_id}-s{self._counts[card_id]}",  # unique across the run
                kind="web",
                locator=result.url,
                title=result.title,
                snippet=result.content[:_MAX_SNIPPET],
            )
            self._by_id[source.source_id] = source
            self._by_card_url[(card_id, result.url)] = source.source_id
            return source

    def get(self, source_id: str) -> Source | None:
        with self._lock:
            return self._by_id.get(source_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)
