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
        revision_searches: int = 2,
        use_documents: bool = False,
    ) -> None:
        self._model = model
        self._system = SystemMessage(content=system_prompt)
        self._tools = tools
        self._sources = sources
        self._audit = audit
        self._max_searches = max_searches
        self._revision_searches = revision_searches
        self._use_documents = use_documents

    def research(self, card: Card) -> Work:
        revision = self._revision_note(card)
        # A revision targets named objections, so it needs fewer searches than the first pass.
        budget = (
            min(self._max_searches, self._revision_searches) if revision else self._max_searches
        )
        kept = card.notes[: card.verified_notes]  # already judged supported: final
        try:
            found = self._search(card, self._queries(card, revision, budget))
            if revision and not found:  # nothing new to write notes from
                raise ResearchFailed("no new sources found for the removed notes")
            candidates = {s.source_id: s for s in [*card.sources, *found]}  # keep earlier sources
            if not candidates:
                raise ResearchFailed("no usable search results")
            draft = self._notes(card, list(candidates.values()), revision)
            return self._to_work(card, draft, candidates)
        except ResearchFailed:
            if not kept:
                raise
            # Nothing new was found, but the verified notes still stand; the Critic settles it.
            return self._work_from(card, kept, {})

    @staticmethod
    def _revision_note(card: Card) -> str:
        if not card.open_objections:
            return ""
        kept = "\n".join(f"- {n.text}" for n in card.notes[: card.verified_notes]) or "(none)"
        removed = "\n".join(f"- {d}" for d in card.dropped) or "(none)"
        problems = "\n".join(f"- {o}" for o in card.open_objections)
        return (
            "\nA reviewer checked your earlier notes against the pages they cite.\n"
            f"Notes that passed. They are final, so do not write them again:\n{kept}\n"
            f"Notes that failed and were removed. Never write these again unchanged:\n{removed}\n"
            f"Problems the reviewer found:\n{problems}\n"
            "Write only NEW notes: a corrected note that says just what a source supports, or a "
            "note backed by better evidence. If a claim cannot be supported, leave it out.\n"
        )

    def _queries(self, card: Card, revision: str, limit: int) -> list[str]:
        request = (
            f"Sub-question: {card.sub_question}\n{revision}\n"
            f"Propose at most {limit} distinct web search queries, fewer if fewer "
            f"are enough. " + json_instruction(QueryPlan)
        )
        plan = self._model.structured([self._system, HumanMessage(content=request)], QueryPlan)
        return plan.queries[:limit]  # the per-Researcher search limit

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
        failed = {d.rsplit(" (", 1)[0].strip().lower() for d in card.dropped}
        have = {n.text.strip().lower() for n in card.notes[: card.verified_notes]}
        notes: list[Note] = []
        for n in draft.notes:
            ids = [i for i in n.source_ids if i in known]
            if len(ids) != len(n.source_ids):
                self._audit.record(
                    "researcher", "unknown_citation_dropped", {"card": card.id, "note": n.text[:80]}
                )
            text = n.text.strip().lower()
            if text in failed or text in have:  # a failed or repeated note adds nothing
                self._audit.record("researcher", "repeat_dropped", {"card": card.id})
                continue
            if ids:
                notes.append(Note(text=n.text, source_ids=ids))
        kept = card.notes[: card.verified_notes]
        if not notes and not kept:
            raise ResearchFailed("the notes cited no known sources")
        scores = {sid: min(1.0, max(0.0, score)) for sid, score in draft.credibility.items()}
        return self._work_from(card, [*kept, *notes], known, scores)

    @staticmethod
    def _work_from(
        card: Card,
        notes: list[Note],
        known: dict[str, Source],
        scores: dict[str, float] | None = None,
    ) -> Work:
        """Notes plus exactly the sources they cite, with ratings carried over from the card."""
        pool = {s.source_id: s for s in [*card.sources, *known.values()]}
        cited = {i for n in notes for i in n.source_ids}
        ratings = {**card.credibility, **(scores or {})}
        return Work(
            notes=notes,
            sources=[s for sid, s in pool.items() if sid in cited],
            credibility={sid: r for sid, r in ratings.items() if sid in cited},
        )
