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


def _card_with_three_notes() -> Board:
    srcs = [
        Source(source_id=f"c1-s{i}", kind="web", locator=f"https://x.example/{i}")
        for i in (1, 2, 3)
    ]
    notes = [Note(text=f"claim {i}", source_ids=[f"c1-s{i}"]) for i in (1, 2, 3)]
    return apply(
        board(),
        [
            CreateCard(op_id="o1", card_id="c1", sub_question="q?"),
            Claim(op_id="o2", card_id="c1"),
            SubmitWork(
                op_id="o3",
                card_id="c1",
                notes=notes,
                sources=srcs,
                credibility={"c1-s1": 0.9, "c1-s2": 0.5, "c1-s3": 0.4},
            ),
        ],
    )


def test_revise_removes_flagged_notes_and_freezes_the_rest() -> None:
    b = _card_with_three_notes()
    v = verdict(
        VerdictKind.REVISE,
        objections=["Note 2 overstates"],
        dropped_notes={2: "partial: overstated"},
    )
    c = apply(b, [Review(op_id="r1", card_id="c1", verdict=v)]).cards["c1"]
    assert c.status is CardStatus.IN_PROGRESS and c.attempts == 2
    assert [n.text for n in c.notes] == ["claim 1", "claim 3"] and c.verified_notes == 2
    assert [s.source_id for s in c.sources] == ["c1-s1", "c1-s3"]
    assert c.credibility == {"c1-s1": 0.9, "c1-s3": 0.4}
    assert c.dropped == ["claim 2 (partial: overstated)"] and c.open_objections == [
        "Note 2 overstates"
    ]


def test_approve_can_drop_notes_and_clears_open_objections() -> None:
    b = _card_with_three_notes()
    v = verdict(VerdictKind.APPROVE, dropped_notes={3: "unsupported: not in the page"})
    c = apply(b, [Review(op_id="r1", card_id="c1", verdict=v)]).cards["c1"]
    assert c.status is CardStatus.DONE and [n.text for n in c.notes] == ["claim 1", "claim 2"]
    assert c.open_objections == [] and c.dropped == ["claim 3 (unsupported: not in the page)"]


def test_cannot_approve_with_every_note_dropped_or_drop_a_missing_note() -> None:
    from research_assistant.board.reducer import BoardError

    b = _card_with_three_notes()
    everything = verdict(VerdictKind.APPROVE, dropped_notes={1: "a", 2: "b", 3: "c"})
    with pytest.raises(BoardError, match="no surviving notes"):
        apply(b, [Review(op_id="r1", card_id="c1", verdict=everything)])
    with pytest.raises(BoardError, match="does not exist"):
        apply(
            b,
            [
                Review(
                    op_id="r2",
                    card_id="c1",
                    verdict=verdict(VerdictKind.REVISE, objections=["x"], dropped_notes={9: "x"}),
                )
            ],
        )
    with pytest.raises(Exception, match="start at 1"):
        verdict(VerdictKind.REVISE, objections=["x"], dropped_notes={0: "x"})


def test_dropped_claims_are_listed_as_limitations() -> None:
    from research_assistant.reporting import limitations

    b = _card_with_three_notes()
    v = verdict(VerdictKind.APPROVE, dropped_notes={3: "unsupported"})
    done = apply(b, [Review(op_id="r1", card_id="c1", verdict=v)])
    assert limitations(done) == ["c1: dropped after failing review: claim 3 (unsupported)"]
