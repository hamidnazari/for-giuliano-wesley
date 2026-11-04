"""The Researcher: searches the web for one card and returns cited notes."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from research_assistant.agents.factory import AgentModel, json_instruction
from research_assistant.agents.protocols import Work
from research_assistant.guardrails.untrusted import wrap_untrusted
from research_assistant.models import Card, Note, Source
from research_assistant.observability.audit import AuditLog
from research_assistant.reliability.retry import RetryableError
from research_assistant.sources import SourceStore
from research_assistant.tools.registry import ToolArgumentRejected, Toolbox
from research_assistant.tools.search import SearchResult


class ResearchFailed(Exception):
    """Nothing usable was found; the card is blocked with this reason."""


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(min_length=1, description="Plain-text web search queries.")


class DraftNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)


class ResearchDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: list[DraftNote] = Field(min_length=1)
    credibility: dict[str, float] = Field(
        default={}, description="Source ID to a 0..1 credibility rating."
    )


class LLMResearcher:
    def __init__(
        self,
        model: AgentModel,
        system_prompt: str,
        tools: Toolbox,
        sources: SourceStore,
        audit: AuditLog,
        max_searches: int = 6,
        use_documents: bool = False,
    ) -> None:
        self._model = model
        self._system = SystemMessage(content=system_prompt)
        self._tools = tools
        self._sources = sources
        self._audit = audit
        self._max_searches = max_searches
        self._use_documents = use_documents

    def research(self, card: Card) -> Work:
        revision = self._revision_note(card)
        found = self._search(card, self._queries(card, revision))
        candidates = {s.source_id: s for s in [*card.sources, *found]}  # keep earlier sources
        if not candidates:
            raise ResearchFailed("no usable search results")
        draft = self._notes(card, list(candidates.values()), revision)
        return self._to_work(card, draft, candidates)

    @staticmethod
    def _revision_note(card: Card) -> str:
        if not card.critic_feedback:
            return ""
        earlier = "\n".join(f"- {n.text} [{', '.join(n.source_ids)}]" for n in card.notes)
        objections = "\n".join(f"- {o}" for o in card.critic_feedback)
        return (
            "\nA reviewer rejected your earlier work. Address these objections.\n"
            f"Objections:\n{objections}\nYour earlier notes:\n{earlier}\n"
        )

    def _queries(self, card: Card, revision: str) -> list[str]:
        request = (
            f"Sub-question: {card.sub_question}\n{revision}\n"
            f"Propose at most {self._max_searches} distinct web search queries, fewer if fewer "
            f"are enough. " + json_instruction(QueryPlan)
        )
        plan = self._model.structured([self._system, HumanMessage(content=request)], QueryPlan)
        return plan.queries[: self._max_searches]  # the per-Researcher search limit

    def _search(self, card: Card, queries: list[str]) -> list[Source]:
        found: dict[str, Source] = {}
        for query in queries:
            try:
                raw = self._tools.call("web_search", query=query)
            except (ToolArgumentRejected, RetryableError, TimeoutError) as e:
                self._audit.record(
                    "researcher", "query_skipped", {"query": query, "error": repr(e)}
                )
                continue
            for item in raw:
                source = self._sources.add(card.id, SearchResult.model_validate(item))
                found[source.source_id] = source
            if self._use_documents:  # local and free, so it does not count as a search
                for hit in self._tools.call("docs_retrieve", query=query):
                    source = self._sources.add_doc(
                        card.id,
                        hit["chunk_id"],
                        hit["doc"] + (f" (page {hit['page']})" if hit["page"] else ""),
                        hit["text"],
                    )
                    found[source.source_id] = source
        return list(found.values())

    def _notes(self, card: Card, sources: list[Source], revision: str) -> ResearchDraft:
        listing = "\n\n".join(
            f"[{s.source_id}] {s.title or '(untitled)'} ({s.locator})\n{s.snippet}" for s in sources
        )
        data = wrap_untrusted(listing, "web search results and excerpts from the user's documents")
        request = (
            f"Sub-question: {card.sub_question}\n{revision}\nSources:\n{data}\n\n"
            "Write the notes now. " + json_instruction(ResearchDraft)
        )
        return self._model.structured([self._system, HumanMessage(content=request)], ResearchDraft)

    def _to_work(self, card: Card, draft: ResearchDraft, known: dict[str, Source]) -> Work:
        notes: list[Note] = []
        for n in draft.notes:
            ids = [i for i in n.source_ids if i in known]
            if len(ids) != len(n.source_ids):
                self._audit.record(
                    "researcher", "unknown_citation_dropped", {"card": card.id, "note": n.text[:80]}
                )
            if ids:
                notes.append(Note(text=n.text, source_ids=ids))
        if not notes:
            raise ResearchFailed("the notes cited no known sources")
        cited = {i for n in notes for i in n.source_ids}
        credibility = {
            sid: min(1.0, max(0.0, score))
            for sid, score in draft.credibility.items()
            if sid in cited
        }
        return Work(
            notes=notes,
            sources=[s for sid, s in known.items() if sid in cited],
            credibility=credibility,
        )
