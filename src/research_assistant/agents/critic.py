"""The Critic: checks every note against the pages it cites, then rules on the card.

The page text is fetched by the system through fetch_cited_passage, never by the Researcher.
Quotes are checked mechanically; the model only judges whether the passages entail each claim.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from research_assistant.agents.factory import AgentModel, json_instruction
from research_assistant.guardrails.untrusted import wrap_untrusted
from research_assistant.models import Card, CitationCheck, Verdict, VerdictKind
from research_assistant.observability.audit import AuditLog
from research_assistant.reliability.retry import RetryableError
from research_assistant.tools.passages import (
    UnknownSource,
    best_windows,
    missing_numbers,
    missing_quotes,
)
from research_assistant.tools.registry import Toolbox

Label = Literal["supported", "partial", "unsupported", "contradicted"]


class NoteJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(description="The note number as shown, starting at 1.")
    label: Label
    reason: str = ""


class CriticReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: list[NoteJudgement]
    source_ratings: dict[str, float] = {}
    recommendation: Literal["approve", "revise", "reject", "split"]
    objections: list[str] = Field(
        default=[],
        description="Card-level problems only. Never repeat a problem already given for a note.",
    )
    split_into: list[str] = []


class LLMCritic:
    def __init__(
        self,
        model: AgentModel,
        system_prompt: str,
        tools: Toolbox,
        audit: AuditLog,
        rubric_version: str,
        final_round: int = 3,
        weak_below: float = 0.3,
    ) -> None:
        self._model = model
        self._system = SystemMessage(content=system_prompt)
        self._tools = tools
        self._audit = audit
        self._rubric_version = rubric_version
        self._weak_below = weak_below
        self._final_round = final_round  # the last review a card gets: it must be settled then

    def review(self, card: Card, round_number: int) -> Verdict:
        first_new = card.verified_notes  # earlier survivors were already judged supported
        new_notes = list(enumerate(card.notes, start=1))[first_new:]
        cited = sorted({sid for _, n in new_notes for sid in n.source_ids})
        passages = self._fetch(cited)
        mechanical: dict[int, list[str]] = {}  # note index -> problems found without the model
        hints: dict[int, list[str]] = {}
        blocks: list[str] = []
        for i, note in new_notes:
            texts: list[str] = []
            for sid in note.source_ids:
                found = passages.get(sid)
                if found is not None:
                    texts.append(found["text"])
            lines = [f"Note {i}: {note.text}", f"Cites: {', '.join(note.source_ids)}"]
            for sid in note.source_ids:
                p = passages.get(sid)
                if not p or not p["text"].strip():
                    lines.append(f"[{sid}] could not be read at all.")
                    continue
                shown = best_windows(p["text"], note.text)
                label = "full page" if p["origin"] == "page" else p["origin"]
                lines.append(f"[{sid}] ({label}) {p['locator']}")
                lines += [f"  passage: {w}" for w in shown]
            if not texts:
                mechanical[i] = ["none of the cited sources could be read"]
            for quote in missing_quotes(note.text, texts) if texts else []:
                mechanical.setdefault(i, []).append(
                    f'quoted text not found in the source: "{quote}"'
                )
            for num in missing_numbers(note.text, texts) if texts else []:
                hints.setdefault(i, []).append(
                    f"the number {num} does not appear in the cited text"
                )
            for text in mechanical.get(i, []) + hints.get(i, []):
                lines.append(f"Mechanical finding: {text}")
            blocks.append("\n".join(lines))
        ratings = "\n".join(
            f"[{s.source_id}] {s.title or s.locator} ({s.kind}); researcher rated "
            f"{card.credibility.get(s.source_id, 'n/a')}"
            for s in card.sources
            if s.source_id in cited
        )
        data = wrap_untrusted("\n\n".join(blocks), "research notes and the pages they cite")
        earlier = ""
        if first_new:
            kept = "\n".join(f"- {n.text}" for n in card.notes[:first_new])
            earlier = (
                f"Notes verified in an earlier round (context only, do not judge):\n{kept}\n\n"
            )
        request = (
            f"Sub-question: {card.sub_question}\nReview round: {round_number}\n\n"
            f"{earlier}{data}\n\n"
            f"Sources to rate:\n{ratings}\n\nJudge every note shown. "
            + json_instruction(CriticReply)
        )
        reply = self._model.structured([self._system, HumanMessage(content=request)], CriticReply)
        return self._rule(card, reply, mechanical, round_number)

    def _fetch(self, source_ids: list[str]) -> dict[str, dict[str, Any] | None]:
        def one(sid: str) -> dict[str, Any] | None:
            try:
                return self._tools.call("fetch_cited_passage", source_id=sid)
            except (UnknownSource, RetryableError, TimeoutError) as e:
                self._audit.record("critic", "fetch_failed", {"source": sid, "error": repr(e)})
                return None

        with ThreadPoolExecutor(max_workers=5) as pool:
            return dict(zip(source_ids, pool.map(one, source_ids), strict=True))

    def _rule(
        self, card: Card, reply: CriticReply, mechanical: dict[int, list[str]], round_number: int
    ) -> Verdict:
        known = {s.source_id for s in card.sources}
        ratings = {
            sid: min(1.0, max(0.0, r)) for sid, r in reply.source_ratings.items() if sid in known
        }
        labels = {j.index: (j.label, j.reason) for j in reply.notes}
        flagged: dict[int, str] = {}  # note number -> why it failed
        for i, note in enumerate(card.notes, start=1):
            if i <= card.verified_notes:
                continue  # judged supported in an earlier round
            label, reason = labels.get(i, ("unsupported", "the Critic did not judge this note"))
            if i in mechanical:  # checked without the model, so the model cannot overrule it
                label, reason = "unsupported", "; ".join(mechanical[i])
            if label != "supported":
                flagged[i] = f"{label}: {reason or 'no reason given'}"
            elif all(
                ratings.get(s, card.credibility.get(s, 0.5)) < self._weak_below
                for s in note.source_ids
            ):
                flagged[i] = f"rests only on weak sources ({', '.join(note.source_ids)})"
        survivors = len(card.notes) - len(flagged)
        final = round_number >= self._final_round
        split_into = [q.strip() for q in reply.split_into if q.strip()]
        kind = VerdictKind(reply.recommendation)
        if kind is VerdictKind.SPLIT and len(split_into) < 2:
            kind = VerdictKind.REVISE
        if kind in (VerdictKind.APPROVE, VerdictKind.REVISE):
            # The evidence decides, not the model's recommendation.
            if not flagged:
                if kind is VerdictKind.REVISE:
                    self._audit.record("critic", "revise_ignored", {"card": card.id})
                kind = VerdictKind.APPROVE
            else:
                if kind is VerdictKind.APPROVE:
                    self._audit.record(
                        "critic", "approve_overridden", {"card": card.id, "notes": sorted(flagged)}
                    )
                # Flagged notes are cut. Keep what survived on the last round, else try again.
                if final:
                    kind = VerdictKind.APPROVE if survivors > 0 else VerdictKind.REJECT
                else:
                    kind = VerdictKind.REVISE
        objections = [
            f"Note {i} ({card.notes[i - 1].text[:140]}) is {why}" for i, why in flagged.items()
        ]
        objections += reply.objections
        if kind in (VerdictKind.REVISE, VerdictKind.REJECT) and not objections:
            objections = ["the evidence does not answer the sub-question"]
        flagged_ids = {s for i in flagged for s in card.notes[i - 1].source_ids}
        checks = [
            CitationCheck(source_id=sid, supported=sid not in flagged_ids)
            for sid in sorted({s for n in card.notes for s in n.source_ids})
        ]
        # An approval lists no unsupported citation; the failed notes are removed instead.
        if kind is VerdictKind.APPROVE:
            checks = [CitationCheck(source_id=c.source_id, supported=True) for c in checks]
        self._audit.record(
            "critic",
            "critic_verdict",
            {
                "card": card.id,
                "round": round_number,
                "kind": kind.value,
                "flagged_notes": sorted(flagged),
            },
        )
        keep_cut = kind in (VerdictKind.APPROVE, VerdictKind.REVISE)
        return Verdict(
            card_id=card.id,
            kind=kind,
            rubric_version=self._rubric_version,
            objections=[] if kind in (VerdictKind.APPROVE, VerdictKind.SPLIT) else objections,
            citation_checks=checks,
            split_into=split_into if kind is VerdictKind.SPLIT else [],
            source_ratings=ratings,
            dropped_notes=flagged if keep_cut else {},
        )
