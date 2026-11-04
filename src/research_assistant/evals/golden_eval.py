"""End-to-end golden run: real agents over a frozen search corpus, scored on a few metrics."""

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from research_assistant.agents.build import agents_builder
from research_assistant.graph.run import RunResult, run_research
from research_assistant.guardrails.budget import Budget
from research_assistant.interaction import AutoApproveIO
from research_assistant.models import CardStatus
from research_assistant.observability.audit import AuditLog
from research_assistant.report_lint import claims, known_sources, lint_report
from research_assistant.settings import Settings
from research_assistant.tools.search import ReplayBackend


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldenQuestion(_Strict):
    id: str
    question: str
    cassette: str


class GoldenThresholds(_Strict):
    min_citation_validity: float = Field(ge=0, le=1)
    min_cards_done_ratio: float = Field(ge=0, le=1)
    max_cost_usd: float = Field(gt=0)
    max_unauthorized_tool_calls: int = Field(ge=0)


@dataclass(frozen=True)
class GoldenMetrics:
    question_id: str
    status: str
    report_written: bool
    claims: int
    citation_validity: float  # share of report claims whose citations resolve to approved sources
    lint_issues: int
    cards: int
    cards_done: int
    unauthorized_tool_calls: int
    cost_usd: float
    llm_calls: int
    tokens: int
    searches: int

    @property
    def cards_done_ratio(self) -> float:
        return self.cards_done / self.cards if self.cards else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cards_done_ratio": round(self.cards_done_ratio, 4)}


def load_questions(path: Path) -> list[GoldenQuestion]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [GoldenQuestion.model_validate(q) for q in data["questions"]]


def load_golden_thresholds(path: Path) -> GoldenThresholds:
    return GoldenThresholds.model_validate(yaml.safe_load(path.read_text())["golden"])


def citation_validity(result: RunResult) -> tuple[int, float]:
    """(claims, share valid). A claim is valid if it cites something and every cited ID is a
    source on an approved card. A report with no claims scores 0, not 1."""
    if not result.report_path:
        return 0, 0.0
    text = Path(result.report_path).read_text(encoding="utf-8")
    ids, _ = known_sources(result.board)
    found = claims(text)
    valid = sum(1 for c in found if c.cited and set(c.cited) <= ids)
    return len(found), (valid / len(found) if found else 0.0)


def measure(question_id: str, result: RunResult, budget: Budget) -> GoldenMetrics:
    n_claims, validity = citation_validity(result)
    issues = (
        lint_report(Path(result.report_path).read_text(encoding="utf-8"), result.board)
        if result.report_path
        else []
    )
    events = AuditLog(result.project_dir / "audit.jsonl", result.run_id).events()
    cards = result.board.cards.values()
    return GoldenMetrics(
        question_id=question_id,
        status=result.status,
        report_written=result.report_path is not None,
        claims=n_claims,
        citation_validity=round(validity, 4),
        lint_issues=len(issues),
        cards=len(result.board.cards),
        cards_done=sum(c.status is CardStatus.DONE for c in cards),
        unauthorized_tool_calls=sum(e.type in {"tool_blocked", "tool_rejected"} for e in events),
        cost_usd=round(budget.total_cost, 4),
        llm_calls=budget.llm_calls,
        tokens=budget.tokens,
        searches=budget.searches,
    )


def check_golden(m: GoldenMetrics, th: GoldenThresholds) -> list[str]:
    out: list[str] = []
    if not m.report_written:
        out.append("no report was written")
    if m.citation_validity < th.min_citation_validity:
        out.append(
            f"citation validity {m.citation_validity:.2f} is below {th.min_citation_validity:.2f}"
        )
    if m.cards_done_ratio < th.min_cards_done_ratio:
        out.append(
            f"cards approved {m.cards_done_ratio:.2f} is below {th.min_cards_done_ratio:.2f}"
        )
    if m.cost_usd > th.max_cost_usd:
        out.append(f"cost ${m.cost_usd:.2f} is above ${th.max_cost_usd:.2f}")
    if m.unauthorized_tool_calls > th.max_unauthorized_tool_calls:
        out.append(f"{m.unauthorized_tool_calls} unauthorized tool call(s)")
    return out


def run_golden(
    q: GoldenQuestion,
    settings: Settings,
    config_dir: Path,
    golden_dir: Path,
    keep: Path | None = None,
) -> GoldenMetrics:
    """keep: a directory to leave the run's board, audit log and report in, for inspection."""
    backend = ReplayBackend(golden_dir / q.cassette, nearest=True)
    make_agents = agents_builder(
        settings,
        config_dir / "prompts",
        real=("planner", "researcher", "critic", "writer"),
        search_backend=backend,
        page_backend=backend,
    )
    budget = Budget(settings.guardrails)
    with tempfile.TemporaryDirectory() as tmp:
        root = keep or Path(tmp)
        result = run_research(
            q.question, settings, root, AutoApproveIO(), make_agents, budget=budget
        )
        return measure(q.id, result, budget)


def format_golden(m: GoldenMetrics) -> str:
    report = "written" if m.report_written else "MISSING"
    return "\n".join(
        [
            f"Golden run '{m.question_id}': {m.status}, report {report}",
            f"  citation validity   {m.citation_validity:.2f}"
            f"  ({m.claims} claims, {m.lint_issues} lint issues)",
            f"  cards approved      {m.cards_done}/{m.cards}",
            f"  unauthorized calls  {m.unauthorized_tool_calls}",
            f"  cost                ${m.cost_usd:.2f}"
            f"  ({m.llm_calls} calls, {m.tokens} tokens, {m.searches} searches)",
        ]
    )
