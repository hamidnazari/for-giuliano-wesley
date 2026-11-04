"""Domain contracts shared by every agent handoff."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CardStatus(StrEnum):
    BACKLOG = "backlog"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    BLOCKED = "blocked"
    REJECTED = "rejected"


class VerdictKind(StrEnum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"
    SPLIT = "split"


class Source(Strict):
    source_id: str = Field(min_length=1)
    kind: Literal["web", "doc"]
    locator: str = Field(min_length=1)  # URL for web, document chunk reference for doc
    title: str = ""
    snippet: str = ""  # stored at retrieval time so the Critic can check against it
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Note(Strict):
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)  # no uncited claims


class Card(Strict):
    id: str = Field(min_length=1)
    sub_question: str = Field(min_length=1)
    status: CardStatus = CardStatus.BACKLOG
    notes: list[Note] = []
    sources: list[Source] = []
    credibility: dict[str, float] = Field(default_factory=dict)  # source_id -> 0..1
    critic_feedback: list[str] = Field(default_factory=list)
    attempts: int = Field(default=0, ge=0)
    parent_id: str | None = None  # set on children created by a split verdict
    reason: str = ""  # why the card is Blocked or Rejected

    @model_validator(mode="after")
    def _check(self) -> Self:
        known = {s.source_id for s in self.sources}
        for note in self.notes:
            missing = set(note.source_ids) - known
            if missing:
                raise ValueError(f"note cites unknown sources: {sorted(missing)}")
        for sid, score in self.credibility.items():
            if sid not in known:
                raise ValueError(f"credibility given for unknown source: {sid}")
            if not 0 <= score <= 1:
                raise ValueError(f"credibility for {sid} must be within 0..1")
        if self.status is CardStatus.DONE and not (self.sources and self.notes):
            raise ValueError("a Done card needs notes and sources")
        if self.status in (CardStatus.BLOCKED, CardStatus.REJECTED) and not self.reason:
            raise ValueError(f"a {self.status.value} card needs a reason")
        return self


class Plan(Strict):
    question: str = Field(min_length=1)
    sub_questions: list[str] = Field(min_length=1)

    def check_cap(self, max_sub_questions: int) -> None:
        if len(self.sub_questions) > max_sub_questions:
            raise ValueError(
                f"plan has {len(self.sub_questions)} sub-questions, cap is {max_sub_questions}"
            )


class CitationCheck(Strict):
    source_id: str
    supported: bool
    note: str = ""


class Verdict(Strict):
    card_id: str
    kind: VerdictKind
    rubric_version: str = Field(min_length=1)
    objections: list[str] = Field(default_factory=list)
    citation_checks: list[CitationCheck] = []
    split_into: list[str] = Field(default_factory=list)  # child sub-questions

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.kind is VerdictKind.SPLIT and not self.split_into:
            raise ValueError("a split verdict needs child sub-questions")
        if self.kind is not VerdictKind.SPLIT and self.split_into:
            raise ValueError("only a split verdict may carry child sub-questions")
        if self.kind in (VerdictKind.REVISE, VerdictKind.REJECT) and not self.objections:
            raise ValueError(f"a {self.kind.value} verdict needs objections")
        if self.kind is VerdictKind.APPROVE and not all(c.supported for c in self.citation_checks):
            raise ValueError("cannot approve a card with an unsupported citation")
        return self


class Board(Strict):
    run_id: str = Field(min_length=1)
    cards: dict[str, Card] = Field(default_factory=dict)
    applied_ops: set[str] = set()  # op ids already applied, so replays are no-ops


class Event(Strict):
    run_id: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str = Field(min_length=1)
    type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
