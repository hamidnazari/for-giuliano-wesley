import pytest
from pydantic import ValidationError

from research_assistant.kanzo_mapping import to_kanzo
from research_assistant.models import (
    Card,
    CardStatus,
    CitationCheck,
    Note,
    Plan,
    Source,
    Verdict,
    VerdictKind,
)

SRC = Source(source_id="s1", kind="web", locator="https://example.com")


def card(**kw: object) -> Card:
    return Card.model_validate({"id": "c1", "sub_question": "q", **kw})


def test_note_must_cite_known_source() -> None:
    with pytest.raises(ValidationError, match="unknown sources"):
        card(sources=[SRC], notes=[Note(text="x", source_ids=["nope"])])


def test_note_requires_a_citation() -> None:
    with pytest.raises(ValidationError):
        Note(text="x", source_ids=[])


def test_done_card_needs_notes_and_sources() -> None:
    with pytest.raises(ValidationError, match="Done card"):
        card(status=CardStatus.DONE)
    ok = card(status=CardStatus.DONE, sources=[SRC], notes=[Note(text="x", source_ids=["s1"])])
    assert ok.status is CardStatus.DONE


@pytest.mark.parametrize("status", [CardStatus.BLOCKED, CardStatus.REJECTED])
def test_blocked_and_rejected_need_reason(status: CardStatus) -> None:
    with pytest.raises(ValidationError, match="needs a reason"):
        card(status=status)
    assert card(status=status, reason="why").reason == "why"


def test_credibility_bounds_and_known_source() -> None:
    with pytest.raises(ValidationError, match="within 0..1"):
        card(sources=[SRC], credibility={"s1": 1.5})
    with pytest.raises(ValidationError, match="unknown source"):
        card(credibility={"s9": 0.5})


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        card(surprise=1)


def test_plan_cap() -> None:
    plan = Plan(question="q", sub_questions=["a", "b", "c"])
    plan.check_cap(3)
    with pytest.raises(ValueError, match="cap is 2"):
        plan.check_cap(2)


def test_verdict_rules() -> None:
    with pytest.raises(ValidationError, match="split"):
        Verdict(card_id="c1", kind=VerdictKind.SPLIT, rubric_version="1")
    with pytest.raises(ValidationError, match="only a split"):
        Verdict(card_id="c1", kind=VerdictKind.APPROVE, rubric_version="1", split_into=["x"])
    with pytest.raises(ValidationError, match="needs objections"):
        Verdict(card_id="c1", kind=VerdictKind.REVISE, rubric_version="1")
    with pytest.raises(ValidationError, match="unsupported citation"):
        Verdict(
            card_id="c1",
            kind=VerdictKind.APPROVE,
            rubric_version="1",
            citation_checks=[CitationCheck(source_id="s1", supported=False)],
        )
    assert Verdict(card_id="c1", kind=VerdictKind.APPROVE, rubric_version="1")


def test_kanzo_mapping_covers_every_status() -> None:
    assert {s: to_kanzo(s) for s in CardStatus} == {
        CardStatus.BACKLOG: ("todo", ()),
        CardStatus.IN_PROGRESS: ("doing", ()),
        CardStatus.IN_REVIEW: ("review", ()),
        CardStatus.DONE: ("done", ()),
        CardStatus.BLOCKED: ("review", ("blocked",)),
        CardStatus.REJECTED: ("review", ("rejected",)),
    }
