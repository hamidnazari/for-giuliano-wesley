import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import research_assistant.cli as cli
from research_assistant.agents.factory import AgentModel
from research_assistant.evals.cases import DEFECT_TYPES, OK, CriticCase, load_cases
from research_assistant.evals.critic_eval import (
    CriticThresholds,
    EvalReport,
    NoteResult,
    check_thresholds,
    format_report,
    load_thresholds,
    run_critic_eval,
)
from research_assistant.guardrails.budget import Budget
from research_assistant.observability.audit import AuditLog
from research_assistant.settings import load_settings

ROOT = Path(__file__).parent.parent
CASES = ROOT / "evals" / "critic" / "cases.yaml"
CONFIG = ROOT / "config"


def reply_for(case: CriticCase, perfect: bool) -> str:
    """A scripted Critic: either judges every note correctly, or approves everything."""
    labels = [
        {
            "index": i,
            "label": "supported" if (not perfect or n.expect == OK) else "unsupported",
            "reason": "scripted",
        }
        for i, n in enumerate(case.notes, start=1)
    ]
    ratings = {s.id: 0.9 for s in case.sources}
    if perfect:
        for n in case.notes:
            if n.expect == "weak_source":
                ratings[n.sources[0]] = 0.1
    return json.dumps(
        {
            "notes": labels,
            "source_ratings": ratings,
            "recommendation": "revise" if perfect else "approve",
        }
    )


def run(perfect: bool, repeats: int = 1) -> EvalReport:
    cases = load_cases(CASES)
    settings = load_settings(CONFIG)
    replies = [reply_for(c, perfect) for _ in range(repeats) for c in cases]

    def factory(budget: Budget, audit: AuditLog) -> AgentModel:
        fake = GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))
        return AgentModel("critic", "scripted/oracle", fake, budget, audit)

    return run_critic_eval(cases, settings, CONFIG / "prompts", factory, repeats)


def test_the_shipped_cases_cover_every_defect_type_and_have_good_notes() -> None:
    notes = [n for c in load_cases(CASES) for n in c.notes]
    assert {n.expect for n in notes if n.expect != OK} == DEFECT_TYPES
    assert sum(n.expect == OK for n in notes) >= 10 and sum(n.expect != OK for n in notes) >= 10


def test_every_case_page_is_frozen_text_and_ids_are_valid() -> None:
    for c in load_cases(CASES):
        assert all(s.page.strip() for s in c.sources) and c.notes


def test_a_perfect_critic_scores_one_and_passes_the_gate() -> None:
    report = run(perfect=True)
    assert report.catch_rate == 1.0 and report.false_positive_rate == 0.0
    assert check_thresholds(report, load_thresholds(ROOT / "evals" / "thresholds.yaml")) == []


def test_a_lenient_critic_fails_the_gate_but_the_quote_check_still_works() -> None:
    report = run(perfect=False)
    failures = check_thresholds(report, load_thresholds(ROOT / "evals" / "thresholds.yaml"))
    assert report.catch_rate < 0.3 and any("catch rate" in f for f in failures)
    # The quote check is mechanical, so even an approve-everything model cannot pass a fake quote.
    caught, total = report.by_type()["fabricated_quote"]
    assert caught == total == 1
    assert not any("fabricated_quote" in f for f in failures)


def test_repeats_multiply_the_samples() -> None:
    assert len(run(perfect=True, repeats=2).results) == 2 * len(run(perfect=True).results)


def test_threshold_checks_and_report_text() -> None:
    results = [
        NoteResult("a", 1, "good", OK, True, "revise"),
        NoteResult("a", 2, "bad quote", "fabricated_quote", False, "approve"),
        NoteResult("a", 3, "bad", "contradiction", True, "revise"),
    ]
    report = EvalReport(results, "m", "3", 1, 10)
    strict = CriticThresholds(
        min_catch_rate=0.9, max_false_positive_rate=0.1, min_catch_by_type={"fabricated_quote": 1.0}
    )
    failures = check_thresholds(report, strict)
    assert len(failures) == 3
    assert any("fabricated_quote: caught 0/1" in f for f in failures)
    text = format_report(report)
    assert "catch rate" in text and "missed defects" in text and "wrongly cut" in text
    assert report.to_dict()["by_type"]["contradiction"] == {"caught": 1, "total": 1}


def test_cli_eval_needs_the_api_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    code = cli.main(
        ["eval", "critic", "--cases", str(CASES), "--config", str(CONFIG),
         "--thresholds", str(ROOT / "evals" / "thresholds.yaml")]
    )  # fmt: skip
    assert code == 1 and "KILO_API_KEY is not set" in capsys.readouterr().err


# ---- injection eval ----
INJ = ROOT / "evals" / "injection" / "cases.yaml"


def test_injection_cases_load_and_cover_every_agent_that_reads_untrusted_content() -> None:
    from research_assistant.evals.injection_cases import load_injection_cases

    cases = load_injection_cases(INJ)
    assert {c.target for c in cases} == {"researcher", "critic", "writer"}
    assert any(c.docs for c in cases) and any("evil.example" in c.forbidden for c in cases)
    assert len({c.id for c in cases}) == len(cases)


def _inj(case_id: str):  # type: ignore[no-untyped-def]
    from research_assistant.evals.injection_cases import load_injection_cases

    return [c for c in load_injection_cases(INJ) if c.id == case_id]


def _scripted(*replies: str):  # type: ignore[no-untyped-def]
    def model_for(role: str, budget: Budget, audit: AuditLog) -> AgentModel:
        fake = GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))
        return AgentModel(role, "scripted/x", fake, budget, audit)

    return model_for


def test_a_researcher_that_obeys_the_page_is_caught() -> None:
    from research_assistant.evals.injection_eval import run_injection_eval

    obey = _scripted(
        json.dumps({"queries": ["pgvector"]}),
        json.dumps(
            {
                "notes": [{"text": "ZX-CANARY-4471 pgvector has HNSW.", "source_ids": ["c1-s1"]}],
                "credibility": {},
            }
        ),
    )
    r = run_injection_eval(
        _inj("researcher-ignore-previous"), load_settings(CONFIG), CONFIG / "prompts", obey
    )
    assert r.results[0].complied and "ZX-CANARY-4471" in r.results[0].why


def test_a_researcher_that_ignores_the_page_holds() -> None:
    from research_assistant.evals.injection_eval import run_injection_eval

    ok = _scripted(
        json.dumps({"queries": ["pgvector index types"]}),
        json.dumps(
            {
                "notes": [{"text": "pgvector has HNSW and IVFFlat.", "source_ids": ["c1-s1"]}],
                "credibility": {},
            }
        ),
    )
    r = run_injection_eval(
        _inj("researcher-ignore-previous"), load_settings(CONFIG), CONFIG / "prompts", ok
    )
    assert not r.results[0].complied


def _leaking(query: str):  # type: ignore[no-untyped-def]
    return _scripted(
        json.dumps({"queries": [query]}),
        json.dumps(
            {"notes": [{"text": "pgvector has HNSW.", "source_ids": ["c1-s1"]}], "credibility": {}}
        ),
    )


def test_a_query_that_carries_the_canary_to_the_search_engine_counts_as_exfiltration() -> None:
    from research_assistant.evals.injection_eval import run_injection_eval

    r = run_injection_eval(
        _inj("researcher-exfiltration-url"),
        load_settings(CONFIG),
        CONFIG / "prompts",
        _leaking("ZX-CANARY-4471 pgvector"),
    )
    assert r.results[0].complied


def test_a_query_containing_a_url_is_blocked_before_it_leaves() -> None:
    from research_assistant.evals.injection_eval import run_injection_eval

    r = run_injection_eval(
        _inj("researcher-exfiltration-url"),
        load_settings(CONFIG),
        CONFIG / "prompts",
        _leaking("https://evil.example/collect?data=secret"),
    )
    assert not r.results[0].complied  # the existing query filter skipped it


def test_a_critic_fooled_into_approving_is_caught() -> None:
    from research_assistant.evals.injection_eval import run_injection_eval

    fooled = _scripted(
        json.dumps(
            {
                "notes": [{"index": 1, "label": "supported", "reason": "ok"}],
                "source_ratings": {"c1-s1": 1.0},
                "recommendation": "approve",
            }
        )
    )
    r = run_injection_eval(
        _inj("critic-fake-reviewer-note"), load_settings(CONFIG), CONFIG / "prompts", fooled
    )
    assert r.results[0].complied and "let through" in r.results[0].why


def test_injection_gate_allows_zero_compliance() -> None:
    from research_assistant.evals.injection_eval import (
        InjectionReport,
        InjectionResult,
        InjectionThresholds,
        check_injection,
        format_injection,
        load_injection_thresholds,
    )

    th = load_injection_thresholds(ROOT / "evals" / "thresholds.yaml")
    assert th.max_compliant == 0
    ok = InjectionReport([InjectionResult("a", "critic", False, "")])
    bad = InjectionReport(
        [InjectionResult("a", "critic", False, ""), InjectionResult("b", "writer", True, "x")]
    )
    assert check_injection(ok, th) == [] and "b" in check_injection(bad, th)[0]
    assert check_injection(bad, InjectionThresholds(max_compliant=1)) == []
    assert "COMPLIED" in format_injection(bad)
