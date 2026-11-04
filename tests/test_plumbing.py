import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_assistant.guardrails.budget import Budget, BudgetExceeded, CardLimitExceeded
from research_assistant.observability.audit import AuditLog, secrets_from_env
from research_assistant.observability.run import new_run_id, project_dir
from research_assistant.settings import GuardrailsConfig


def caps(**kw: int) -> GuardrailsConfig:
    base = {
        "max_llm_calls": 3,
        "max_tokens": 100,
        "max_searches": 2,
        "max_wall_clock_seconds": 10,
        "max_critic_rounds": 2,
        "max_card_attempts": 2,
        "max_sub_questions": 5,
    }
    return GuardrailsConfig(**(base | kw))


def test_run_id_format_and_uniqueness() -> None:
    now = datetime(2026, 11, 5, 7, 15, 30, tzinfo=UTC)
    a, b = new_run_id(now), new_run_id(now)
    assert re.fullmatch(r"20261105-071530-[0-9a-f]{6}", a)
    assert a != b


def test_project_dir_created_once(tmp_path: Path) -> None:
    path = project_dir(tmp_path, "r1")
    assert path == tmp_path / "projects" / "r1" and path.is_dir()
    with pytest.raises(FileExistsError):
        project_dir(tmp_path, "r1")


def test_audit_log_appends_and_reads_back(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a" / "audit.jsonl", "r1", secrets=[])
    log.record("researcher", "tool_call", {"tool": "web_search"})
    log.record("critic", "verdict", {"kind": "approve"})
    events = AuditLog(log.path, "r1", secrets=[]).events()
    assert [e.type for e in events] == ["tool_call", "verdict"]
    assert len(log.path.read_text().splitlines()) == 2


def test_audit_log_redacts_secrets(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", "r1", secrets=["sk-supersecret123"])
    log.record("planner", "note", {"text": "key is sk-supersecret123"})
    text = log.path.read_text()
    assert "sk-supersecret123" not in text and "[REDACTED]" in text
    json.loads(text)


def test_secrets_from_env_picks_key_like_names() -> None:
    fake = "not-a-" + "real-key-value"  # built at runtime so scanners see no literal secret
    env = {"KILO_API_KEY": fake, "PATH": "/usr/bin/long/path", "SHORT_KEY": "abc"}
    assert secrets_from_env(env) == [fake]


def test_llm_call_cap() -> None:
    b = Budget(caps())
    for _ in range(3):
        b.before_llm_call()
        b.record_llm_call("researcher", 5, 5, 0.01)
    with pytest.raises(BudgetExceeded, match="llm_calls"):
        b.before_llm_call()
    assert b.llm_calls == 3 and b.tokens == 30


def test_token_cap_and_cost_per_agent() -> None:
    b = Budget(caps(max_llm_calls=10))
    b.record_llm_call("critic", 60, 50, 0.5)
    b.record_llm_call("writer", 1, 1, 0.25)
    assert b.usage["critic"].cost == 0.5 and b.total_cost == 0.75
    with pytest.raises(BudgetExceeded, match="tokens"):
        b.before_llm_call()


def test_search_cap() -> None:
    b = Budget(caps())
    b.before_search()
    b.before_search()
    with pytest.raises(BudgetExceeded, match="searches"):
        b.before_search()


def test_wall_clock_cap() -> None:
    now = [0.0]
    b = Budget(caps(), clock=lambda: now[0])
    b.before_llm_call()
    now[0] = 11.0
    with pytest.raises(BudgetExceeded, match="wall_clock"):
        b.before_llm_call()


def test_card_limits_are_per_card_and_not_run_level() -> None:
    b = Budget(caps())
    assert b.start_attempt("c1") == 1 and b.start_attempt("c1") == 2
    with pytest.raises(CardLimitExceeded, match="attempts"):
        b.start_attempt("c1")
    assert b.start_attempt("c2") == 1
    b.start_critic_round("c1")
    b.start_critic_round("c1")
    with pytest.raises(CardLimitExceeded, match="critic_rounds"):
        b.start_critic_round("c1")
