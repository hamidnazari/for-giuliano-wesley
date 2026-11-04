"""Deterministic stand-ins for the real agents. They let the whole graph run with no model."""

from collections.abc import Mapping, Sequence

from research_assistant.agents.protocols import Work
from research_assistant.guardrails.budget import Budget
from research_assistant.interaction import UserIO
from research_assistant.models import (
    Board,
    Card,
    CitationCheck,
    Note,
    Plan,
    Source,
    Verdict,
    VerdictKind,
)
from research_assistant.reporting import plain_report


class StubPlanner:
    def __init__(self, sub_questions: int = 2) -> None:
        self._n = sub_questions

    def plan(self, question: str, io: UserIO) -> Plan:
        return Plan(
            question=question,
            sub_questions=[f"{question} (aspect {i})" for i in range(1, self._n + 1)],
        )


class StubResearcher:
    def __init__(self, budget: Budget | None = None) -> None:
        self._budget = budget

    def research(self, card: Card) -> Work:
        if self._budget:
            self._budget.before_search()  # a stand-in for the real search charge
        source = Source(
            source_id=f"{card.id}-s1",
            kind="web",
            locator=f"https://example.com/{card.id}",
            title=f"Stub source for {card.id}",
            snippet=f"Stub finding for {card.sub_question}",
        )
        note = Note(text=f"Stub finding for: {card.sub_question}", source_ids=[source.source_id])
        return Work(notes=[note], sources=[source], credibility={source.source_id: 0.8})


class StubCritic:
    """Approves by default. script maps a card id to the verdict kinds for rounds 1, 2, ..."""

    def __init__(self, script: Mapping[str, Sequence[VerdictKind]] | None = None) -> None:
        self._script = script or {}

    def review(self, card: Card, round_number: int) -> Verdict:
        kinds = self._script.get(card.id, ())
        kind = kinds[round_number - 1] if round_number <= len(kinds) else VerdictKind.APPROVE
        match kind:
            case VerdictKind.APPROVE:
                checks = [
                    CitationCheck(source_id=s.source_id, supported=True) for s in card.sources
                ]
                return Verdict(
                    card_id=card.id, kind=kind, rubric_version="stub", citation_checks=checks
                )
            case VerdictKind.SPLIT:
                return Verdict(
                    card_id=card.id,
                    kind=kind,
                    rubric_version="stub",
                    split_into=[f"{card.sub_question} (part {i})" for i in (1, 2)],
                )
            case _:
                return Verdict(
                    card_id=card.id, kind=kind, rubric_version="stub", objections=["stub objection"]
                )


class StubWriter:
    def write(self, board: Board, question: str, stop_reason: str | None) -> str:
        return plain_report(board, question, stop_reason)
