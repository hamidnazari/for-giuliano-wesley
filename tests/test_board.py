from pathlib import Path

import pytest

from research_assistant.board.ops import (
    Block,
    Claim,
    CreateCard,
    Reject,
    Review,
    SubmitWork,
)
from research_assistant.board.reducer import (
    CardCapExceeded,
    DuplicateCard,
    InvalidTransition,
    UnknownCard,
    apply,
)
from research_assistant.board.render import render_board
from research_assistant.board.sink import InMemorySink
from research_assistant.models import (
    Board,
    CardStatus,
    CitationCheck,
    Note,
    Source,
    Verdict,
    VerdictKind,
)

SRC = Source(source_id="s1", kind="web", locator="https://example.com")
NOTE = Note(text="a fact", source_ids=["s1"])


def board() -> Board:
    return Board(run_id="r1")


def verdict(kind: VerdictKind, card_id: str = "c1", **kw: object) -> Verdict:
    return Verdict.model_validate({"card_id": card_id, "kind": kind, "rubric_version": "1", **kw})


def to_review(b: Board, card_id: str = "c1", n: int = 1) -> Board:
    return apply(
        b,
        [
            CreateCard(op_id=f"create:{card_id}", card_id=card_id, sub_question="q?"),
            Claim(op_id=f"claim:{card_id}:{n}", card_id=card_id),
            SubmitWork(op_id=f"submit:{card_id}:{n}", card_id=card_id, notes=[NOTE], sources=[SRC]),
        ],
    )


def test_happy_path_to_done() -> None:
    b = to_review(board())
    assert b.cards["c1"].status is CardStatus.IN_REVIEW and b.cards["c1"].attempts == 1
    ok = verdict(
        VerdictKind.APPROVE, citation_checks=[CitationCheck(source_id="s1", supported=True)]
    )
    b = apply(b, [Review(op_id="review:c1:1", card_id="c1", verdict=ok)])
    assert b.cards["c1"].status is CardStatus.DONE


def test_revise_loops_back_with_feedback_and_counts_attempt() -> None:
    b = to_review(board())
    rev = verdict(VerdictKind.REVISE, objections=["weak source"])
    b = apply(b, [Review(op_id="review:c1:1", card_id="c1", verdict=rev)])
    c = b.cards["c1"]
    assert c.status is CardStatus.IN_PROGRESS and c.attempts == 2
    assert c.critic_feedback == ["weak source"]


def test_reject_records_reason() -> None:
    b = to_review(board())
    rej = verdict(VerdictKind.REJECT, objections=["fake citation", "contradiction"])
    b = apply(b, [Review(op_id="r1", card_id="c1", verdict=rej)])
    assert b.cards["c1"].status is CardStatus.REJECTED
    assert b.cards["c1"].reason == "fake citation; contradiction"


def test_split_creates_children_and_rejects_parent() -> None:
    b = to_review(board())
    sp = verdict(VerdictKind.SPLIT, split_into=["part a?", "part b?"])
    b = apply(b, [Review(op_id="r1", card_id="c1", verdict=sp)])
    assert b.cards["c1"].status is CardStatus.REJECTED
    assert b.cards["c1"].reason == "split into c1.1, c1.2"
    assert b.cards["c1.1"].parent_id == "c1" and b.cards["c1.1"].status is CardStatus.BACKLOG
    assert b.cards["c1.2"].sub_question == "part b?"


def test_split_counts_against_cap_and_is_atomic() -> None:
    b = to_review(board())
    sp = verdict(VerdictKind.SPLIT, split_into=["a?", "b?"])
    with pytest.raises(CardCapExceeded):
        apply(b, [Review(op_id="r1", card_id="c1", verdict=sp)], max_cards=2)
    assert b.cards["c1"].status is CardStatus.IN_REVIEW and len(b.cards) == 1


def test_create_respects_cap_and_rejects_duplicates() -> None:
    b = apply(board(), [CreateCard(op_id="o1", card_id="c1", sub_question="q")], max_cards=1)
    with pytest.raises(CardCapExceeded):
        apply(b, [CreateCard(op_id="o2", card_id="c2", sub_question="q")], max_cards=1)
    with pytest.raises(DuplicateCard):
        apply(b, [CreateCard(op_id="o3", card_id="c1", sub_question="q")])


def test_replayed_ops_are_noops() -> None:
    ops = [
        CreateCard(op_id="create:c1", card_id="c1", sub_question="q"),
        Claim(op_id="claim:c1:1", card_id="c1"),
    ]
    once = apply(board(), ops)
    twice = apply(once, ops)
    assert twice == once and twice.cards["c1"].attempts == 1


def test_invalid_transitions_rejected() -> None:
    b = apply(board(), [CreateCard(op_id="o1", card_id="c1", sub_question="q")])
    with pytest.raises(InvalidTransition):
        apply(b, [SubmitWork(op_id="o2", card_id="c1", notes=[NOTE], sources=[SRC])])
    done = to_review(board())
    ok = verdict(VerdictKind.APPROVE)
    done = apply(done, [Review(op_id="r", card_id="c1", verdict=ok)])
    with pytest.raises(InvalidTransition):
        apply(done, [Reject(op_id="x", card_id="c1", reason="late")])
    with pytest.raises(UnknownCard):
        apply(b, [Claim(op_id="o9", card_id="nope")])


def test_blocked_column_keeps_reason_and_is_not_dropped() -> None:
    b = apply(
        board(),
        [
            CreateCard(op_id="o1", card_id="c1", sub_question="q"),
            Claim(op_id="o2", card_id="c1"),
            Block(op_id="o3", card_id="c1", reason="search timed out"),
        ],
    )
    assert b.cards["c1"].status is CardStatus.BLOCKED
    assert b.cards["c1"].reason == "search timed out"
    assert "reason: search timed out" in render_board(b)


def test_failed_batch_leaves_input_untouched() -> None:
    b = apply(board(), [CreateCard(op_id="o1", card_id="c1", sub_question="q")])
    before = b.model_copy(deep=True)
    with pytest.raises(UnknownCard):
        apply(
            b,
            [
                CreateCard(op_id="o2", card_id="c2", sub_question="q"),
                Claim(op_id="o3", card_id="missing"),
            ],
        )
    assert b == before


def test_mismatched_verdict_rejected() -> None:
    b = to_review(board())
    with pytest.raises(Exception, match="verdict is for"):
        apply(b, [Review(op_id="r", card_id="c1", verdict=verdict(VerdictKind.APPROVE, "c9"))])


def test_render_groups_by_column() -> None:
    b = to_review(board())
    out = render_board(b)
    assert "In review (1)" in out and "Backlog (0)" in out and "c1  q?" in out
    assert "1 notes, 1 sources, attempts 1" in out


def test_in_memory_sink_records() -> None:
    sink = InMemorySink()
    ops = [CreateCard(op_id="o1", card_id="c1", sub_question="q")]
    b = apply(board(), ops)
    sink.sync(b, ops)
    assert sink.ops == ops and sink.latest == b and sink.latest is not b


def test_snapshot_roundtrip_and_board_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from research_assistant.board.sink import JsonSnapshotSink, load_snapshot
    from research_assistant.cli import main

    ops = [
        CreateCard(op_id="o1", card_id="c1", sub_question="q?"),
        Claim(op_id="o2", card_id="c1"),
        Block(op_id="o3", card_id="c1", reason="timed out"),
    ]
    b = apply(Board(run_id="r1"), ops)
    path = tmp_path / "projects" / "r1" / "board.json"
    JsonSnapshotSink(path).sync(b, ops)
    assert load_snapshot(path) == b
    assert main(["board", "--data", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Blocked (1)" in out and "timed out" in out
    assert main(["board", "--data", str(tmp_path), "--run", "nope"]) == 1
