from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from research_assistant.agents.factory import (
    KILO_BASE_URL,
    MalformedOutput,
    ModelFactory,
)
from research_assistant.guardrails.budget import Budget, BudgetExceeded
from research_assistant.observability.audit import AuditLog
from research_assistant.settings import Settings, load_settings

CONFIG = Path(__file__).parent.parent / "config"


class Answer(BaseModel):
    value: int


def setup(
    tmp_path: Path, scripts: dict[str, list[str]], env: dict[str, str] | None = None
) -> tuple[ModelFactory, Budget, AuditLog, Settings]:
    settings = load_settings(CONFIG)
    budget = Budget(settings.guardrails)
    audit = AuditLog(tmp_path / "audit.jsonl", "r1", secrets=[])
    prices = {"anthropic/claude-opus-4.7": (0.001, 0.002)}
    return (
        ModelFactory(settings, budget, audit, scripts, prices, env or {}),
        budget,
        audit,
        settings,
    )


def test_scripted_model_returns_responses_in_order(tmp_path: Path) -> None:
    factory, budget, audit, _ = setup(tmp_path, {"planner": ["one", "two"]})
    m = factory.build("planner")
    assert m.invoke([HumanMessage(content="hi")]).content == "one"
    assert m.invoke([HumanMessage(content="hi")]).content == "two"
    assert budget.llm_calls == 2 and budget.usage["planner"].llm_calls == 2
    assert [e.type for e in audit.events()] == ["llm_call", "llm_call"]
    assert audit.events()[0].payload["estimated"] is True


def test_cost_uses_model_prices(tmp_path: Path) -> None:
    factory, budget, _, _ = setup(tmp_path, {"critic": ["x" * 40]})
    factory.build("critic").invoke([HumanMessage(content="y" * 40)])
    assert budget.usage["critic"].cost == pytest.approx(10 * 0.001 + 10 * 0.002)


def test_budget_blocks_before_the_call(tmp_path: Path) -> None:
    factory, _, _, settings = setup(tmp_path, {"planner": ["a"] * 500})
    m = factory.build("planner")
    for _ in range(settings.guardrails.max_llm_calls):
        m.invoke([HumanMessage(content="hi")])
    with pytest.raises(BudgetExceeded, match="llm_calls"):
        m.invoke([HumanMessage(content="hi")])


def test_structured_parses_json_and_fences(tmp_path: Path) -> None:
    factory, _, _, _ = setup(tmp_path, {"planner": ['```json\n{"value": 3}\n```']})
    assert factory.build("planner").structured([HumanMessage(content="q")], Answer).value == 3


def test_structured_repairs_malformed_output(tmp_path: Path) -> None:
    factory, budget, audit, _ = setup(
        tmp_path, {"critic": ["not json", '{"value": "x"}', '{"value": 9}']}
    )
    out = factory.build("critic").structured([HumanMessage(content="q")], Answer)
    assert out.value == 9 and budget.llm_calls == 3
    assert [e.type for e in audit.events()].count("malformed_output") == 2


def test_structured_gives_up(tmp_path: Path) -> None:
    factory, _, _, _ = setup(tmp_path, {"critic": ["bad"] * 3})
    with pytest.raises(MalformedOutput):
        factory.build("critic").structured([HumanMessage(content="q")], Answer)


def test_real_model_uses_kilo_gateway(tmp_path: Path) -> None:
    factory, _, _, settings = setup(tmp_path, {}, env={"KILO_API_KEY": "placeholder"})
    m = factory.build("critic")
    inner = m._model  # pyright: ignore[reportPrivateUsage]
    assert isinstance(inner, ChatOpenAI)
    assert inner.openai_api_base == KILO_BASE_URL
    assert inner.model_name == settings.agents.critic.model
    assert inner.max_retries == 0


def test_real_model_needs_api_key(tmp_path: Path) -> None:
    factory, _, _, _ = setup(tmp_path, {}, env={})
    with pytest.raises(RuntimeError, match="KILO_API_KEY"):
        factory.build("planner")


def test_unknown_agent(tmp_path: Path) -> None:
    factory, _, _, _ = setup(tmp_path, {})
    with pytest.raises(ValueError, match="unknown agent"):
        factory.build("reflector")
