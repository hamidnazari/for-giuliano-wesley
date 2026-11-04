"""Run the Critic on planted-defect cards and score how many bad notes it cuts."""

import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from research_assistant.agents.critic import LLMCritic
from research_assistant.agents.factory import AgentModel, ModelFactory
from research_assistant.agents.prompts import read_prompt
from research_assistant.evals.cases import OK, CriticCase
from research_assistant.guardrails.budget import Budget
from research_assistant.models import Card, CardStatus, Note, Source, Verdict, VerdictKind
from research_assistant.observability.audit import AuditLog
from research_assistant.settings import Settings
from research_assistant.sources import SourceStore
from research_assistant.tools.passages import PassageCache, make_fetch_cited_passage
from research_assistant.tools.registry import Tool, ToolRegistry
from research_assistant.tools.search import SearchResult


@dataclass(frozen=True)
class NoteResult:
    case_id: str
    note: int  # 1-based
    text: str
    expect: str
    flagged: bool
    verdict: str


@dataclass
class EvalReport:
    results: list[NoteResult]
    model: str
    rubric_version: str
    llm_calls: int
    tokens: int

    def _defects(self) -> list[NoteResult]:
        return [r for r in self.results if r.expect != OK]

    def _clean(self) -> list[NoteResult]:
        return [r for r in self.results if r.expect == OK]

    @property
    def catch_rate(self) -> float:
        d = self._defects()
        return sum(r.flagged for r in d) / len(d) if d else 1.0

    @property
    def false_positive_rate(self) -> float:
        c = self._clean()
        return sum(r.flagged for r in c) / len(c) if c else 0.0

    def by_type(self) -> dict[str, tuple[int, int]]:
        """Defect type -> (caught, total)."""
        out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in self._defects():
            out[r.expect][0] += int(r.flagged)
            out[r.expect][1] += 1
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "rubric_version": self.rubric_version,
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "catch_rate": round(self.catch_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "by_type": {k: {"caught": c, "total": t} for k, (c, t) in self.by_type().items()},
            "notes": [asdict(r) for r in self.results],
        }


class CriticThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_catch_rate: float = Field(ge=0, le=1)
    max_false_positive_rate: float = Field(ge=0, le=1)
    min_catch_by_type: dict[str, float] = {}


def load_thresholds(path: Path) -> CriticThresholds:
    return CriticThresholds.model_validate(yaml.safe_load(path.read_text())["critic"])


def check_thresholds(report: EvalReport, th: CriticThresholds) -> list[str]:
    """Human-readable failures. Empty means the gate passes."""
    failures: list[str] = []
    if report.catch_rate < th.min_catch_rate:
        failures.append(f"catch rate {report.catch_rate:.2f} is below {th.min_catch_rate:.2f}")
    if report.false_positive_rate > th.max_false_positive_rate:
        failures.append(
            f"false positive rate {report.false_positive_rate:.2f} is above "
            f"{th.max_false_positive_rate:.2f}"
        )
    by_type = report.by_type()
    for kind, minimum in th.min_catch_by_type.items():
        caught, total = by_type.get(kind, (0, 0))
        if total and caught / total < minimum:
            failures.append(f"{kind}: caught {caught}/{total}, needs {minimum:.2f}")
    return failures


class _Pages:
    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages

    def extract(self, url: str) -> str:
        return self._pages.get(url, "")


def _noop(**_: Any) -> list[Any]:
    return []


def _flagged_notes(verdict: Verdict, count: int) -> set[int]:
    if verdict.kind is VerdictKind.REJECT:
        return set(range(1, count + 1))  # the whole card was thrown out
    if verdict.kind is VerdictKind.SPLIT:
        return set()  # the Critic did not rule on the notes
    return set(verdict.dropped_notes)


def build_case(
    case: CriticCase,
    model: AgentModel,
    settings: Settings,
    system_prompt: str,
    budget: Budget,
    audit: AuditLog,
) -> tuple[LLMCritic, Card]:
    store = SourceStore()
    ids: dict[str, str] = {}
    for s in case.sources:
        added = store.add("c1", SearchResult(url=s.url, title=s.title, content=s.page[:300]))
        ids[s.id] = added.source_id
    pages = _Pages({s.url: s.page for s in case.sources})
    registry = ToolRegistry(settings, audit)
    fetch = make_fetch_cited_passage(store, pages, PassageCache(), budget)
    registry.register(Tool("fetch_cited_passage", fetch))
    for name in ("web_search", "render_report"):
        registry.register(Tool(name, _noop))
    sources: list[Source] = []
    for s in case.sources:
        found = store.get(ids[s.id])
        assert found is not None
        sources.append(found)
    card = Card(
        id="c1",
        sub_question=case.sub_question,
        status=CardStatus.IN_REVIEW,
        notes=[Note(text=n.text, source_ids=[ids[i] for i in n.sources]) for n in case.notes],
        sources=sources,
        credibility={src.source_id: 0.8 for src in sources},  # the Researcher trusted them all
    )
    critic = LLMCritic(
        model,
        system_prompt,
        registry.for_agent("critic"),
        audit,
        settings.agents.rubric_version,
        final_round=99,  # score the first-round verdict: flagged notes are cut, never approved
    )
    return critic, card


def run_critic_eval(
    cases: list[CriticCase],
    settings: Settings,
    prompts_dir: Path,
    factory: Callable[[Budget, AuditLog], AgentModel] | None = None,
    repeats: int = 1,
) -> EvalReport:
    budget = Budget(settings.guardrails.model_copy(update={"max_llm_calls": 10_000}))
    with tempfile.TemporaryDirectory() as tmp:
        audit = AuditLog(Path(tmp) / "audit.jsonl", "eval")
        model = (
            factory(budget, audit)
            if factory
            else ModelFactory(settings, budget, audit).build("critic")
        )
        prompt = (
            read_prompt(prompts_dir, settings.agents.critic.prompt)
            + "\n\n"
            + read_prompt(prompts_dir, settings.agents.critic_rubric)
        )
        results: list[NoteResult] = []
        for _ in range(repeats):
            for case in cases:
                critic, card = build_case(case, model, settings, prompt, budget, audit)
                verdict = critic.review(card, 1)
                flagged = _flagged_notes(verdict, len(case.notes))
                for i, note in enumerate(case.notes, start=1):
                    results.append(
                        NoteResult(
                            case.id, i, note.text, note.expect, i in flagged, verdict.kind.value
                        )
                    )
    return EvalReport(
        results, model.model_id, settings.agents.rubric_version, budget.llm_calls, budget.tokens
    )


def format_report(report: EvalReport) -> str:
    lines = [
        f"Critic defect eval: model {report.model}, rubric {report.rubric_version}, "
        f"{report.llm_calls} calls, {report.tokens} tokens",
        f"  catch rate           {report.catch_rate:.2f}",
        f"  false positive rate  {report.false_positive_rate:.2f}",
        "  by defect type:",
    ]
    lines += [f"    {k:18} {c}/{t}" for k, (c, t) in report.by_type().items()]
    misses = [r for r in report.results if r.expect != OK and not r.flagged]
    wrong = [r for r in report.results if r.expect == OK and r.flagged]
    if misses:
        lines.append("  missed defects:")
        lines += [f"    {r.case_id} note {r.note} ({r.expect}): {r.text[:90]}" for r in misses]
    if wrong:
        lines.append("  good notes wrongly cut:")
        lines += [f"    {r.case_id} note {r.note}: {r.text[:90]}" for r in wrong]
    return "\n".join(lines)
