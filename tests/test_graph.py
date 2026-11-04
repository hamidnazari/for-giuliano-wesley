import shutil
from pathlib import Path

import pytest

from research_assistant.agents.env import AgentEnv
from research_assistant.agents.protocols import Agents
from research_assistant.agents.stubs import (
    StubCritic,
    StubPlanner,
    StubResearcher,
    StubWriter,
)
from research_assistant.cli import main
from research_assistant.graph.run import RunResult, run_research
from research_assistant.interaction import AutoApproveIO, ScriptedIO
from research_assistant.models import CardStatus, VerdictKind
from research_assistant.observability.audit import AuditLog
from research_assistant.settings import Settings, load_settings

CONFIG = Path(__file__).parent.parent / "config"
V = VerdictKind


def settings(**caps: int) -> Settings:
    s = load_settings(CONFIG)
    return s.model_copy(update={"guardrails": s.guardrails.model_copy(update=caps)})


def go(
    tmp_path: Path,
    s: Settings | None = None,
    sub_questions: int = 2,
    script: dict[str, list[VerdictKind]] | None = None,
    io: ScriptedIO | AutoApproveIO | None = None,
    **kw: object,
) -> RunResult:
    def make(env: AgentEnv) -> Agents:
        return Agents(
            StubPlanner(sub_questions), StubResearcher(env.budget), StubCritic(script), StubWriter()
        )

    return run_research("What is X?", s or settings(), tmp_path, io or AutoApproveIO(), make, **kw)  # type: ignore[arg-type]


def statuses(r: RunResult) -> dict[str, CardStatus]:
    return {c.id: c.status for c in r.board.cards.values()}


def event_types(r: RunResult) -> list[str]:
    return [e.type for e in AuditLog(r.project_dir / "audit.jsonl", r.run_id, secrets=[]).events()]


def test_happy_path_writes_a_report_from_approved_cards(tmp_path: Path) -> None:
    r = go(tmp_path, sub_questions=3)
    assert r.status == "completed"
    assert set(statuses(r).values()) == {CardStatus.DONE}
    report = Path(r.report_path or "").read_text()
    assert report.startswith("# What is X?")
    assert "aspect 1" in report and "aspect 3" in report and "## Sources" in report
    assert "Limitations" not in report
    assert (r.project_dir / "board.json").exists()
    types = event_types(r)
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert {"plan_created", "plan_approval", "board_op", "tool_call", "report_written"} <= set(
        types
    )


def test_declined_plan_spends_nothing(tmp_path: Path) -> None:
    r = go(tmp_path, io=ScriptedIO(approve=False))
    assert r.status == "declined" and r.report_path is None and not r.board.cards
    assert "board_op" not in event_types(r)


def test_revise_then_approve(tmp_path: Path) -> None:
    r = go(tmp_path, sub_questions=1, script={"c1": [V.REVISE, V.APPROVE]})
    assert statuses(r) == {"c1": CardStatus.DONE}
    assert r.board.cards["c1"].attempts == 3 - 1  # claim + one revise
    assert r.board.cards["c1"].critic_feedback == ["stub objection"]


def test_critic_rounds_exhausted_blocks_card_and_report_admits_gap(tmp_path: Path) -> None:
    r = go(tmp_path, settings(max_critic_rounds=2), 2, {"c1": [V.REVISE] * 5})
    assert statuses(r) == {"c1": CardStatus.BLOCKED, "c2": CardStatus.DONE}
    assert "needs your decision" in r.board.cards["c1"].reason
    report = Path(r.report_path or "").read_text()
    assert "## Limitations" in report and "c1:" in report and "blocked" in report


def test_attempt_limit_rejects_card(tmp_path: Path) -> None:
    r = go(tmp_path, settings(max_card_attempts=2, max_critic_rounds=5), 1, {"c1": [V.REVISE] * 5})
    assert statuses(r) == {"c1": CardStatus.REJECTED}
    assert "attempts limit" in r.board.cards["c1"].reason


def test_split_children_are_researched_and_parent_is_not_a_gap(tmp_path: Path) -> None:
    r = go(tmp_path, sub_questions=1, script={"c1": [V.SPLIT]})
    assert statuses(r) == {
        "c1": CardStatus.REJECTED,
        "c1.1": CardStatus.DONE,
        "c1.2": CardStatus.DONE,
    }
    assert "Limitations" not in Path(r.report_path or "").read_text()


def test_split_over_the_cap_becomes_a_rejection(tmp_path: Path) -> None:
    r = go(tmp_path, settings(max_sub_questions=2), 2, {"c1": [V.SPLIT]})
    assert statuses(r)["c1"] is CardStatus.REJECTED
    assert set(statuses(r)) == {"c1", "c2"}
    assert "exceed the sub-question cap" in r.board.cards["c1"].reason


def test_budget_cap_stops_cleanly_with_a_partial_report(tmp_path: Path) -> None:
    r = go(tmp_path, settings(max_searches=1), sub_questions=3)
    assert r.status == "stopped" and "searches" in (r.stop_reason or "")
    report = Path(r.report_path or "").read_text()
    assert "partial" in report and "not completed" in report
    assert "run_stopped" in event_types(r)


def test_step_limit_stops_cleanly(tmp_path: Path) -> None:
    r = go(tmp_path, sub_questions=1, recursion_limit=4)
    assert r.status == "stopped" and "step limit" in (r.stop_reason or "")
    assert r.report_path and Path(r.report_path).exists()


def test_plan_over_sub_question_cap_fails_before_spend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cap is 2"):
        go(tmp_path, settings(max_sub_questions=2), sub_questions=3)


@pytest.mark.skipif(shutil.which("kanzo") is None, reason="kanzo is not installed")
def test_run_mirrors_into_kanzo(tmp_path: Path) -> None:
    r = go(tmp_path, sub_questions=2, kanzo=True, kanzo_home=tmp_path / "kanzo")
    assert r.status == "completed"
    assert "sink_error" not in event_types(r)


def test_cli_run_auto_approve(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["run", "What is X?", "--auto-approve", "--config", str(CONFIG), "--data", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0 and "completed" in out and "report:" in out and "Done (2)" in out


def test_cli_run_declined_at_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("builtins.input", lambda _="": "n")
    code = main(["run", "What is X?", "--config", str(CONFIG), "--data", str(tmp_path)])
    assert code == 0 and "declined" in capsys.readouterr().out


def test_cli_run_eof_at_the_prompt_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def eof(_: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert main(["run", "q", "--config", str(CONFIG), "--data", str(tmp_path)]) == 0
    assert "declined" in capsys.readouterr().out
