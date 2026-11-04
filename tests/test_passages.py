from pathlib import Path
from typing import Any

import pytest

from research_assistant.guardrails.budget import Budget, BudgetExceeded
from research_assistant.observability.audit import AuditLog
from research_assistant.settings import load_settings
from research_assistant.sources import SourceStore
from research_assistant.tools.passages import (
    PassageCache,
    UnknownSource,
    best_windows,
    make_fetch_cited_passage,
    missing_numbers,
    missing_quotes,
)
from research_assistant.tools.registry import Tool, ToolRegistry
from research_assistant.tools.search import SearchResult

CONFIG = Path(__file__).parent.parent / "config"


class Pages:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def extract(self, url: str) -> str:
        self.calls.append(url)
        return self.pages.get(url, "")


def setup(pages: dict[str, str], max_fetches: int = 100):
    settings = load_settings(CONFIG)
    g = settings.guardrails.model_copy(update={"max_page_fetches": max_fetches})
    budget = Budget(g)
    store = SourceStore()
    web = store.add("c1", SearchResult(url="https://a.example/1", title="A", content="snippet A"))
    web2 = store.add("c1", SearchResult(url="https://b.example/2", title="B", content="snippet B"))
    doc = store.add_doc("c1", "notes.md#c1", "notes.md", "chunk text of notes")
    backend = Pages(pages)
    fetch = make_fetch_cited_passage(store, backend, PassageCache(), budget)
    return fetch, backend, budget, (web.source_id, web2.source_id, doc.source_id)


def test_fetches_the_full_page_once_and_caches_it() -> None:
    fetch, backend, budget, (web, _, _) = setup({"https://a.example/1": "the full page text"})
    first = fetch(source_id=web)
    again = fetch(source_id=web)
    assert first["text"] == "the full page text" and first["origin"] == "page"
    assert again == first and backend.calls == ["https://a.example/1"] and budget.fetches == 1


def test_unreadable_page_falls_back_to_the_search_snippet() -> None:
    fetch, _, _, (_, web2, _) = setup({})
    out = fetch(source_id=web2)
    assert out["origin"] == "snippet" and out["text"] == "snippet B"


def test_document_sources_use_the_stored_chunk_and_cost_nothing() -> None:
    fetch, backend, budget, (_, _, doc) = setup({})
    out = fetch(source_id=doc)
    assert out["origin"] == "document" and out["text"] == "chunk text of notes"
    assert backend.calls == [] and budget.fetches == 0


def test_unknown_source_and_fetch_cap() -> None:
    fetch, _, _, (web, web2, _) = setup({"https://a.example/1": "x", "https://b.example/2": "y"}, 1)
    with pytest.raises(UnknownSource):
        fetch(source_id="nope-s1")
    fetch(source_id=web)
    with pytest.raises(BudgetExceeded, match="page_fetches"):
        fetch(source_id=web2)


def test_the_critic_cannot_reach_the_tool_with_a_url(tmp_path: Path) -> None:
    from research_assistant.tools.registry import ToolArgumentRejected

    settings = load_settings(CONFIG)
    fetch, _, _, (web, _, _) = setup({"https://a.example/1": "text"})
    audit = AuditLog(tmp_path / "a.jsonl", "r1", secrets=[])
    reg = ToolRegistry(settings, audit)

    def noop(**_: Any) -> list[Any]:
        return []

    for name in ("web_search", "docs_retrieve", "render_report"):
        reg.register(Tool(name, noop))
    reg.register(Tool("fetch_cited_passage", fetch))
    critic = reg.for_agent("critic")
    assert critic.call("fetch_cited_passage", source_id=web)["text"] == "text"
    with pytest.raises(ToolArgumentRejected):
        critic.call("fetch_cited_passage", source_id="https://evil.example/x")


def test_quotes_must_appear_verbatim_modulo_whitespace_and_case() -> None:
    page = "He said:  “Latency grew past the 150 ms budget”  at scale."
    note = 'The team wrote "latency grew past the 150 ms budget" in testing.'
    assert missing_quotes(note, [page]) == []
    fake = 'They also claimed "performance was always excellent everywhere".'
    assert missing_quotes(fake, [page]) == ["performance was always excellent everywhere"]
    assert missing_quotes("no quotes here", [page]) == []
    assert missing_quotes('short "tiny" quote', [page]) == []  # under 12 characters is ignored


def test_numbers_are_checked_loosely() -> None:
    page = "Sufficient up to 5,000,000 vectors; latency 150 ms; cost $50/month and 75% less."
    assert missing_numbers("Up to 5000000 vectors at 150 ms.", [page]) == []
    assert missing_numbers("It is 42% faster and costs $50.", [page]) == ["42"]
    assert missing_numbers("Latency of 50 ms.", ["latency was 150 ms"]) == ["50"]  # not inside 150
    assert missing_numbers("About 5 vectors.", ["It has 5.5 vectors."]) == ["5"]


def test_best_windows_pick_the_relevant_stretch() -> None:
    filler = "Lorem ipsum dolor sit amet. " * 200
    page = (
        filler + "The pgvector extension supports HNSW and IVFFlat indexes for vectors. " + filler
    )
    got = best_windows(page, "pgvector supports HNSW and IVFFlat indexes", count=1, size=400)
    assert len(got) == 1 and "HNSW" in got[0]
    assert (
        best_windows("short page", "anything") == ["short page"] and best_windows("  ", "x") == []
    )
