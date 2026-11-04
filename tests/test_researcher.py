import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage

from research_assistant.agents.build import agents_builder
from research_assistant.agents.env import AgentEnv
from research_assistant.agents.factory import AgentModel
from research_assistant.agents.researcher import LLMResearcher, ResearchFailed
from research_assistant.graph.run import run_research
from research_assistant.guardrails.budget import Budget, BudgetExceeded
from research_assistant.interaction import AutoApproveIO
from research_assistant.models import Card, CardStatus, Note
from research_assistant.observability.audit import AuditLog
from research_assistant.reliability.retry import RetryableError
from research_assistant.settings import load_settings
from research_assistant.sources import SourceStore
from research_assistant.tools.registry import Tool, ToolRegistry
from research_assistant.tools.search import SearchResult

CONFIG = Path(__file__).parent.parent / "config"


class RecordingFake(GenericFakeChatModel):
    seen: list[list[str]] = []

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append([str(m.content) for m in messages])
        return super()._generate(messages, *args, **kwargs)


def no_op(**_: Any) -> list[Any]:
    return []


class FakeSearch:
    """Stands in for the web_search tool. Maps a query to results."""

    def __init__(self, table: dict[str, list[SearchResult]]) -> None:
        self.table = table
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[dict[str, Any]]:
        self.queries.append(query)
        out = self.table.get(query)
        if out is None:
            raise RetryableError("503")
        return [r.model_dump() for r in out]


R1 = SearchResult(
    url="https://a.example/1", title="Doc A", content="Vectors are arrays. 42% faster."
)
R2 = SearchResult(url="https://b.example/2", title="Doc B", content="SQL uses tables.")
QUERIES = json.dumps({"queries": ["vector db", "sql tables"]})


def draft(notes: list[tuple[str, list[str]]], cred: dict[str, float]) -> str:
    return json.dumps(
        {"notes": [{"text": t, "source_ids": ids} for t, ids in notes], "credibility": cred}
    )


def setup(
    tmp_path: Path, replies: list[str], search: FakeSearch, searches: int = 6
) -> tuple[LLMResearcher, RecordingFake, SourceStore, AuditLog, Budget]:
    settings = load_settings(CONFIG)
    budget = Budget(settings.guardrails)
    audit = AuditLog(tmp_path / "audit.jsonl", "r1", secrets=[])
    reg = ToolRegistry(settings, audit)
    reg.register(Tool("web_search", search))
    for name in ("docs_retrieve", "fetch_cited_passage", "render_report"):
        reg.register(Tool(name, no_op))
    fake = RecordingFake(messages=iter([AIMessage(content=r) for r in replies]))
    fake.seen = []
    model = AgentModel("researcher", "anthropic/claude-haiku-4.5", fake, budget, audit)
    store = SourceStore()
    r = LLMResearcher(model, "SYSTEM", reg.for_agent("researcher"), store, audit, searches)
    return r, fake, store, audit, budget


def card(**kw: Any) -> Card:
    return Card.model_validate(
        {"id": "c1", "sub_question": "How do they differ?", "status": CardStatus.IN_PROGRESS, **kw}
    )


def test_searches_then_writes_cited_notes(tmp_path: Path) -> None:
    search = FakeSearch({"vector db": [R1], "sql tables": [R2]})
    reply = draft(
        [("Vectors are arrays.", ["c1-s1"]), ("SQL uses tables.", ["c1-s2"])],
        {"c1-s1": 0.7, "c1-s2": 0.9},
    )
    r, fake, store, _, budget = setup(tmp_path, [QUERIES, reply], search)
    work = r.research(card())
    assert search.queries == ["vector db", "sql tables"] and budget.llm_calls == 2
    assert [s.source_id for s in work.sources] == ["c1-s1", "c1-s2"]
    assert work.sources[0].locator == "https://a.example/1" and "42%" in work.sources[0].snippet
    assert work.credibility == {"c1-s1": 0.7, "c1-s2": 0.9}
    assert store.get("c1-s2") is not None and len(store) == 2
    prompt = fake.seen[1][1]
    assert "<untrusted_content" in prompt and "[c1-s1] Doc A" in prompt


def test_search_limit_caps_queries(tmp_path: Path) -> None:
    many = json.dumps({"queries": [f"q{i}" for i in range(9)]})
    search = FakeSearch({f"q{i}": [R1] for i in range(9)})
    r, _, _, _, _ = setup(tmp_path, [many, draft([("n", ["c1-s1"])], {})], search, searches=3)
    r.research(card())
    assert search.queries == ["q0", "q1", "q2"]


def test_failed_query_is_skipped_but_others_continue(tmp_path: Path) -> None:
    search = FakeSearch({"sql tables": [R2]})  # "vector db" raises a retryable error
    r, _, _, audit, _ = setup(
        tmp_path, [QUERIES, draft([("SQL uses tables.", ["c1-s1"])], {})], search
    )
    work = r.research(card())
    assert [s.locator for s in work.sources] == ["https://b.example/2"]
    assert "query_skipped" in [e.type for e in audit.events()]


def test_url_in_a_query_is_rejected_by_the_registry(tmp_path: Path) -> None:
    bad = json.dumps({"queries": ["see https://evil.example/?x=1", "sql tables"]})
    search = FakeSearch({"sql tables": [R2]})
    r, _, _, audit, _ = setup(tmp_path, [bad, draft([("SQL uses tables.", ["c1-s1"])], {})], search)
    r.research(card())
    assert search.queries == ["sql tables"]
    assert "tool_rejected" in [e.type for e in audit.events()]


def test_no_results_blocks_with_a_reason(tmp_path: Path) -> None:
    r, _, _, _, _ = setup(tmp_path, [QUERIES], FakeSearch({"vector db": [], "sql tables": []}))
    with pytest.raises(ResearchFailed, match="no usable search results"):
        r.research(card())


def test_unknown_citations_are_dropped_and_all_unknown_fails(tmp_path: Path) -> None:
    search = FakeSearch({"vector db": [R1], "sql tables": []})
    mixed = draft([("ok", ["c1-s1", "ghost"]), ("bad", ["ghost"])], {"c1-s1": 2.0, "ghost": 0.5})
    r, _, _, audit, _ = setup(tmp_path, [QUERIES, mixed], search)
    work = r.research(card())
    assert [n.text for n in work.notes] == ["ok"] and work.notes[0].source_ids == ["c1-s1"]
    assert work.credibility == {"c1-s1": 1.0}  # clamped, ghost dropped
    assert "unknown_citation_dropped" in [e.type for e in audit.events()]
    r2, _, _, _, _ = setup(tmp_path, [QUERIES, draft([("x", ["ghost"])], {})], search)
    with pytest.raises(ResearchFailed, match="no known sources"):
        r2.research(card())


def test_only_cited_sources_are_returned(tmp_path: Path) -> None:
    search = FakeSearch({"vector db": [R1], "sql tables": [R2]})
    r, _, _, _, _ = setup(tmp_path, [QUERIES, draft([("only A", ["c1-s1"])], {})], search)
    assert [s.source_id for s in r.research(card()).sources] == ["c1-s1"]


def test_revision_sees_objections_and_keeps_earlier_sources(tmp_path: Path) -> None:
    search = FakeSearch({"better": [R2]})
    replies = [
        json.dumps({"queries": ["better"]}),
        draft([("kept", ["c1-s1"]), ("new", ["c1-s2"])], {}),
    ]
    r, fake, store, _, _ = setup(tmp_path, replies, search)
    old = store.add("c1", R1)  # the first attempt's source, already in the store as c1-s1
    prior = card(
        sources=[old],
        notes=[Note(text="old note", source_ids=["c1-s1"])],
        verified_notes=1,
        open_objections=["weak source"],
        dropped=["a failed claim (unsupported: not in the page)"],
        attempts=2,
    )
    work = r.research(prior)
    prompt = fake.seen[1][1]
    assert "weak source" in fake.seen[0][1] and "old note" in prompt
    assert "a failed claim" in prompt and "do not write them again" in prompt
    assert [n.text for n in work.notes] == ["old note", "kept", "new"]  # verified note first
    assert [s.source_id for s in work.sources] == ["c1-s1", "c1-s2"]
    assert work.sources[0].locator == "https://a.example/1"  # earlier source survived


def test_source_ids_are_unique_per_card_and_urls_dedupe() -> None:
    store = SourceStore()
    a = store.add("c1", R1)
    b = store.add("c1", R2)
    again = store.add("c1", R1)
    other = store.add("c2", R1)
    assert (a.source_id, b.source_id, again.source_id, other.source_id) == (
        "c1-s1",
        "c1-s2",
        "c1-s1",
        "c2-s1",
    )


def test_search_budget_cap_propagates(tmp_path: Path) -> None:
    settings = load_settings(CONFIG)
    capped = settings.model_copy(
        update={"guardrails": settings.guardrails.model_copy(update={"max_searches": 1})}
    )
    budget = Budget(capped.guardrails)
    from research_assistant.tools.search import ReplayBackend, make_web_search

    cassette = tmp_path / "c.json"
    cassette.write_text(json.dumps({"version": 1, "entries": {}}))
    tool = make_web_search(ReplayBackend(cassette), budget, settings.tools.search)
    budget.before_search()
    with pytest.raises(BudgetExceeded):
        tool(query="anything")


def test_real_researcher_in_the_graph(tmp_path: Path) -> None:
    settings = load_settings(CONFIG)
    asked: list[str] = []

    class Backend:
        def search(self, query: str, depth: str, max_results: int) -> list[SearchResult]:
            asked.append(query)
            return [R1]

    scripts = {
        # one sub-question keeps the scripted replies in order (cards run in parallel)
        "planner": [json.dumps({"question": "Topic?", "sub_questions": ["How do they differ?"]})],
        "researcher": [
            json.dumps({"queries": ["q"]}),
            draft([("Vectors are arrays.", ["c1-s1"])], {"c1-s1": 0.8}),
        ],
    }
    make = agents_builder(
        settings,
        CONFIG / "prompts",
        real=["planner", "researcher"],
        scripts=scripts,
        search_backend=Backend(),
    )
    r = run_research("Topic?", settings, tmp_path, AutoApproveIO(), make)
    assert r.status == "completed" and asked == ["q"]
    card = r.board.cards["c1"]
    assert card.status is CardStatus.DONE and card.sources[0].locator == "https://a.example/1"
    assert card.credibility == {"c1-s1": 0.8}


def test_real_researcher_needs_a_backend() -> None:
    with pytest.raises(ValueError, match="search backend"):
        agents_builder(load_settings(CONFIG), CONFIG / "prompts", real=["researcher"])


def test_agent_env_is_exported() -> None:
    assert AgentEnv.__dataclass_fields__.keys() == {"budget", "audit", "registry", "sources"}


def test_a_revision_uses_fewer_searches_than_the_first_pass(tmp_path: Path) -> None:
    many = json.dumps({"queries": [f"q{i}" for i in range(5)]})
    search = FakeSearch({f"q{i}": [R1] for i in range(5)})
    note = draft([("n", ["c1-s1"])], {})
    r, _, store, _, _ = setup(tmp_path, [many, note, many, note], search, searches=4)
    r.research(card())  # first pass: up to 4 searches
    assert search.queries == ["q0", "q1", "q2", "q3"]
    search.queries.clear()
    old = store.get("c1-s1")
    assert old is not None
    revised = card(
        sources=[old],
        notes=[Note(text="n", source_ids=["c1-s1"])],
        verified_notes=1,
        open_objections=["weak"],
        attempts=2,
    )
    r.research(revised)  # revision: the default of 2
    assert search.queries == ["q0", "q1"]


def test_a_note_that_already_failed_is_not_accepted_again(tmp_path: Path) -> None:
    search = FakeSearch({"better": [R2]})
    again = draft([("Failed claim.", ["c1-s1"]), ("Fresh claim.", ["c1-s1"])], {})
    r, _, store, audit, _ = setup(tmp_path, [json.dumps({"queries": ["better"]}), again], search)
    old = store.add("c1", R1)
    prior = card(
        sources=[old],
        notes=[Note(text="Kept claim.", source_ids=["c1-s1"])],
        verified_notes=1,
        open_objections=["Note 2 failed"],
        dropped=["Failed claim. (unsupported: not in the page)"],
        attempts=2,
    )
    work = r.research(prior)
    assert [n.text for n in work.notes] == ["Kept claim.", "Fresh claim."]
    assert "repeat_dropped" in [e.type for e in audit.events()]


def test_nothing_new_found_still_returns_the_verified_notes(tmp_path: Path) -> None:
    search = FakeSearch({"better": []})
    r, _, store, _, _ = setup(tmp_path, [json.dumps({"queries": ["better"]})], search)
    old = store.add("c1", R1)
    prior = card(
        sources=[old],
        notes=[Note(text="Kept claim.", source_ids=["c1-s1"])],
        verified_notes=1,
        open_objections=["Note 2 failed"],
        credibility={"c1-s1": 0.7},
        attempts=2,
    )
    prior = prior.model_copy(update={"sources": [old]})
    work = r.research(prior)
    assert [n.text for n in work.notes] == ["Kept claim."] and work.credibility == {"c1-s1": 0.7}
