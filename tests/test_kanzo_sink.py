import shutil
from pathlib import Path

import pytest

from research_assistant.board.kanzo import KanzoSink, SafeSink
from research_assistant.board.ops import Block, Claim, CreateCard, Review, SubmitWork
from research_assistant.board.reducer import apply
from research_assistant.models import (
    Board,
    CitationCheck,
    Note,
    Source,
    Verdict,
    VerdictKind,
)
from research_assistant.observability.audit import AuditLog

pytestmark = pytest.mark.skipif(shutil.which("kanzo") is None, reason="kanzo is not installed")

SLUG = "20261105-071530-a1b2c3"
SRC = Source(source_id="s1", kind="web", locator="https://example.com", title="Example")


def step(sink: KanzoSink, board: Board, *ops: object) -> Board:
    new = apply(board, list(ops))  # type: ignore[arg-type]
    sink.sync(new, list(ops))  # type: ignore[arg-type]
    return new


def test_card_lifecycle_is_mirrored(tmp_path: Path) -> None:
    with KanzoSink(SLUG, home=tmp_path / "kanzo") as sink:
        b = Board(run_id="r1")
        b = step(sink, b, CreateCard(op_id="o1", card_id="c1", sub_question="What is X?"))
        task = sink.task_id("c1")
        assert task == f"{SLUG}#1"
        assert "status: todo" in sink.call("get_task", id=task)

        b = step(sink, b, Claim(op_id="o2", card_id="c1"))
        assert "status: doing" in sink.call("get_task", id=task)

        b = step(
            sink,
            b,
            SubmitWork(
                op_id="o3",
                card_id="c1",
                notes=[Note(text="X is Y", source_ids=["s1"])],
                sources=[SRC],
            ),
        )
        text = sink.call("get_task", id=task)
        assert "status: review" in text and "X is Y" in text and "researcher" in text

        ok = Verdict(
            card_id="c1",
            kind=VerdictKind.APPROVE,
            rubric_version="1",
            citation_checks=[CitationCheck(source_id="s1", supported=True)],
        )
        b = step(sink, b, Review(op_id="o4", card_id="c1", verdict=ok))
        text = sink.call("get_task", id=task)
        assert "status: done" in text and "Verdict: approve" in text and "critic" in text


def test_blocked_is_a_tag_on_review_with_reason(tmp_path: Path) -> None:
    with KanzoSink(SLUG, home=tmp_path / "kanzo") as sink:
        b = Board(run_id="r1")
        b = step(sink, b, CreateCard(op_id="o1", card_id="c1", sub_question="q?"))
        b = step(sink, b, Claim(op_id="o2", card_id="c1"))
        b = step(sink, b, Block(op_id="o3", card_id="c1", reason="search timed out"))
        text = sink.call("get_task", id=sink.task_id("c1"))
        assert "{blocked}" in text and "status: review" in text
        assert "search timed out" in text


def test_split_creates_child_tasks(tmp_path: Path) -> None:
    with KanzoSink(SLUG, home=tmp_path / "kanzo") as sink:
        b = Board(run_id="r1")
        b = step(
            sink,
            b,
            CreateCard(op_id="o1", card_id="c1", sub_question="big?"),
            Claim(op_id="o2", card_id="c1"),
            SubmitWork(
                op_id="o3", card_id="c1", notes=[Note(text="n", source_ids=["s1"])], sources=[SRC]
            ),
        )
        sp = Verdict(
            card_id="c1", kind=VerdictKind.SPLIT, rubric_version="1", split_into=["a?", "b?"]
        )
        b = step(sink, b, Review(op_id="o4", card_id="c1", verdict=sp))
        assert sink.task_id("c1.1") != sink.task_id("c1.2")
        listing = sink.call("list_tasks", project=SLUG, status="all")
        assert "c1.1: a?" in listing and "c1.2: b?" in listing
        assert "{rejected}" in sink.call("get_task", id=sink.task_id("c1"))


def test_safe_sink_logs_failures_instead_of_raising(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl", "r1", secrets=[])
    sink = KanzoSink(SLUG, home=tmp_path / "kanzo")  # never opened: calls will fail
    op = CreateCard(op_id="o1", card_id="c1", sub_question="q")
    board = apply(Board(run_id="r1"), [op])
    SafeSink(sink, audit).sync(board, [op])
    assert [e.type for e in audit.events()] == ["sink_error"]
