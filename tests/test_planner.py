import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage

from research_assistant.agents.build import agents_builder
from research_assistant.agents.factory import AgentModel, MalformedOutput
from research_assistant.agents.planner import LLMPlanner
from research_assistant.cli import main
from research_assistant.graph.run import run_research
from research_assistant.guardrails.budget import Budget
from research_assistant.interaction import AutoApproveIO, ScriptedIO
from research_assistant.models import CardStatus
from research_assistant.observability.audit import AuditLog
from research_assistant.settings import load_settings

CONFIG = Path(__file__).parent.parent / "config"


class RecordingFake(GenericFakeChatModel):
    """A scripted model that also remembers the prompts it was given."""

    seen: list[list[str]] = []

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append([str(m.content) for m in messages])
        return super()._generate(messages, *args, **kwargs)


def planner(
    tmp_path: Path, replies: Sequence[object], cap: int = 4, clarifying: int = 3
) -> tuple[LLMPlanner, RecordingFake, Budget]:
    settings = load_settings(CONFIG)
    budget = Budget(settings.guardrails)
    audit = AuditLog(tmp_path / "audit.jsonl", "r1", secrets=[])
    fake = RecordingFake(
        messages=iter(
            [AIMessage(content=r if isinstance(r, str) else json.dumps(r)) for r in replies]
        )
    )
    fake.seen = []
    model = AgentModel("planner", "anthropic/claude-sonnet-4.6", fake, budget, audit)
    return LLMPlanner(model, "SYSTEM PROMPT", cap, clarifying), fake, budget


PLAN = {"question": "Refined?", "sub_questions": ["a?", "b?"]}


def test_clear_question_asks_nothing_and_plans(tmp_path: Path) -> None:
    p, fake, budget = planner(tmp_path, [{"questions": []}, PLAN])
    io = ScriptedIO()
    plan = p.plan("What is X?", io)
    assert plan.sub_questions == ["a?", "b?"] and plan.question == "Refined?"
    assert io.asked == [] and budget.llm_calls == 2
    assert fake.seen[0][0] == "SYSTEM PROMPT"


def test_clarifying_answers_reach_the_plan_prompt(tmp_path: Path) -> None:
    p, fake, _ = planner(tmp_path, [{"questions": ["Which region?", "Which year?"]}, PLAN])
    io = ScriptedIO(["Australia", ""])  # second answer is blank and is skipped
    p.plan("Housing prices?", io)
    assert io.asked == ["Which region?", "Which year?"]
    plan_prompt = fake.seen[1][1]
    assert "Q: Which region?\nA: Australia" in plan_prompt and "Which year?" not in plan_prompt


def test_clarifying_questions_are_capped(tmp_path: Path) -> None:
    qs = [f"q{i}?" for i in range(6)]
    p, _, _ = planner(tmp_path, [{"questions": qs}, PLAN], clarifying=2)
    io = ScriptedIO(["x", "y", "z"])
    p.plan("Q", io)
    assert io.asked == ["q0?", "q1?"]


def test_non_interactive_skips_the_clarify_call(tmp_path: Path) -> None:
    p, _, budget = planner(tmp_path, [PLAN])
    p.plan("Q", AutoApproveIO())
    assert budget.llm_calls == 1


def test_too_many_sub_questions_are_repaired(tmp_path: Path) -> None:
    too_many = {"question": "Q", "sub_questions": [f"s{i}?" for i in range(9)]}
    p, _, budget = planner(tmp_path, [too_many, PLAN], cap=4)
    assert len(p.plan("Q", AutoApproveIO()).sub_questions) == 2
    assert budget.llm_calls == 2


def test_unfixable_output_raises(tmp_path: Path) -> None:
    p, _, _ = planner(tmp_path, ["nope"] * 3)
    with pytest.raises(MalformedOutput):
        p.plan("Q", AutoApproveIO())


def test_real_planner_inside_the_graph(tmp_path: Path) -> None:
    settings = load_settings(CONFIG)
    scripts = {
        "planner": [
            json.dumps({"questions": ["Which region?"]}),
            json.dumps({"question": "Refined?", "sub_questions": ["a?", "b?", "c?"]}),
        ]
    }
    make = agents_builder(settings, CONFIG / "prompts", real=["planner"], scripts=scripts)
    r = run_research("Q", settings, tmp_path, ScriptedIO(["AU"]), make)
    assert r.status == "completed"
    assert {c.sub_question for c in r.board.cards.values()} == {"a?", "b?", "c?"}
    assert all(c.status is CardStatus.DONE for c in r.board.cards.values())
    assert r.board.cards["c1"].sub_question == "a?"


def test_unknown_real_agent_rejected(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no real implementation"):
        agents_builder(load_settings(CONFIG), CONFIG / "prompts", real=["critic"])


def test_cli_real_planner_needs_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    argv = ["run", "Q", "--auto-approve", "--real", "planner"]
    code = main([*argv, "--config", str(CONFIG), "--data", str(tmp_path)])
    assert code == 1 and "KILO_API_KEY is not set" in capsys.readouterr().err
