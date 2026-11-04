from pathlib import Path
from typing import Any

import pytest

from research_assistant.guardrails.budget import Budget, BudgetExceeded
from research_assistant.observability.audit import AuditLog
from research_assistant.settings import SearchConfig, load_settings
from research_assistant.tools.registry import Tool, ToolRegistry
from research_assistant.tools.search import (
    RecordingBackend,
    ReplayBackend,
    ReplayMiss,
    SearchResult,
    TavilyBackend,
    make_backend,
    make_web_search,
)

CONFIG = Path(__file__).parent.parent / "config"


class StubTavily:
    pages: dict[str, str] = {"https://a.example/1": "Full page text about vectors."}

    def extract(self, urls: list[str], **kwargs: Any) -> dict[str, Any]:
        return {
            "results": [{"url": u, "raw_content": self.pages[u]} for u in urls if u in self.pages],
            "failed_results": [{"url": u} for u in urls if u not in self.pages],
        }

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((query, kwargs))
        return {
            "results": [
                {
                    "url": "https://a.example",
                    "title": "A",
                    "content": "alpha",
                    "score": 0.9,
                    "x": 1,
                },
                {"url": "https://b.example"},
            ]
        }


def test_tavily_backend_maps_results_and_passes_settings() -> None:
    stub = StubTavily()
    results = TavilyBackend(stub).search("Rust GC", "advanced", 3)
    assert stub.calls == [("Rust GC", {"search_depth": "advanced", "max_results": 3})]
    assert results[0] == SearchResult(
        url="https://a.example", title="A", content="alpha", score=0.9
    )
    assert results[1].title == "" and results[1].score == 0.0


def test_record_then_replay_roundtrip(tmp_path: Path) -> None:
    cassette = tmp_path / "c" / "search.json"
    recorded = RecordingBackend(TavilyBackend(StubTavily()), cassette).search(
        "Rust  GC", "basic", 5
    )
    replayed = ReplayBackend(cassette).search("rust gc", "basic", 5)  # case and spacing ignored
    assert replayed == recorded and len(replayed) == 2


def test_replay_miss_fails_loudly(tmp_path: Path) -> None:
    cassette = tmp_path / "search.json"
    RecordingBackend(TavilyBackend(StubTavily()), cassette).search("known", "basic", 5)
    replay = ReplayBackend(cassette)
    with pytest.raises(ReplayMiss):
        replay.search("unknown", "basic", 5)
    with pytest.raises(ReplayMiss):
        replay.search("known", "advanced", 5)  # depth is part of the key


def test_replay_never_needs_a_key(tmp_path: Path) -> None:
    cassette = tmp_path / "search.json"
    RecordingBackend(TavilyBackend(StubTavily()), cassette).search("q", "basic", 5)
    assert make_backend("replay", cassette, env={}).search("q", "basic", 5)


def test_live_and_record_need_a_key_and_record_needs_a_cassette(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        make_backend("live", env={})
    with pytest.raises(ValueError, match="cassette"):
        make_backend("record", None, env={})
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        make_backend("record", tmp_path / "c.json", env={})


def test_web_search_charges_budget_and_stops_at_cap(tmp_path: Path) -> None:
    settings = load_settings(CONFIG)
    budget = Budget(settings.guardrails.model_copy(update={"max_searches": 2}))
    cassette = tmp_path / "s.json"
    RecordingBackend(TavilyBackend(StubTavily()), cassette).search("q", "basic", 5)
    tool = make_web_search(ReplayBackend(cassette), budget, SearchConfig())
    tool(query="q")
    tool(query="q")
    with pytest.raises(BudgetExceeded, match="searches"):
        tool(query="q")
    assert budget.searches == 2


def noop(**_: Any) -> None:
    return None


def test_web_search_through_registry_for_researcher_only(tmp_path: Path) -> None:
    settings = load_settings(CONFIG)
    cassette = tmp_path / "s.json"
    RecordingBackend(TavilyBackend(StubTavily()), cassette).search("q", "basic", 5)
    fn = make_web_search(ReplayBackend(cassette), Budget(settings.guardrails), SearchConfig())
    audit = AuditLog(tmp_path / "audit.jsonl", "r1", secrets=[])
    reg = ToolRegistry(settings, audit)
    for name in ("web_search", "docs_retrieve", "fetch_cited_passage", "render_report"):
        reg.register(Tool(name, fn if name == "web_search" else noop))
    out = reg.for_agent("researcher").call("web_search", query="q")
    assert out[0]["url"] == "https://a.example"


def test_extract_through_tavily_record_and_replay(tmp_path: Path) -> None:
    cassette = tmp_path / "s.json"
    rec = RecordingBackend(TavilyBackend(StubTavily()), cassette)
    assert rec.extract("https://a.example/1") == "Full page text about vectors."
    assert rec.extract("https://nope.example") == ""  # unreadable page: empty, not an error
    rec.search("q", "basic", 5)  # recording a search keeps the recorded pages
    replay = ReplayBackend(cassette)
    assert replay.extract("https://a.example/1") == "Full page text about vectors."
    assert len(replay.search("q", "basic", 5)) == 2
    with pytest.raises(ReplayMiss):
        replay.extract("https://never-recorded.example")
