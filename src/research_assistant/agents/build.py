"""Pick real or stub agents for a run. Agents not yet built are always stubs."""

from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path

from research_assistant.agents.factory import ModelFactory
from research_assistant.agents.planner import LLMPlanner
from research_assistant.agents.prompts import read_prompt
from research_assistant.agents.protocols import Agents, Writer
from research_assistant.agents.stubs import StubCritic, StubPlanner, StubResearcher, StubWriter
from research_assistant.agents.writer import LLMWriter, ReportChecker
from research_assistant.guardrails.budget import Budget
from research_assistant.observability.audit import AuditLog
from research_assistant.settings import Settings

REAL_AGENTS = ("planner", "writer")  # grows as real agents are built


def agents_builder(
    settings: Settings,
    prompts_dir: Path,
    real: Collection[str] = (),
    scripts: Mapping[str, Sequence[str]] | None = None,
) -> Callable[[Budget, AuditLog], Agents]:
    """Return make_agents(budget, audit). scripts feed fake models in tests."""
    unknown = set(real) - set(REAL_AGENTS)
    if unknown:
        raise ValueError(f"no real implementation yet for: {sorted(unknown)}")

    def make(budget: Budget, audit: AuditLog) -> Agents:
        models = ModelFactory(settings, budget, audit, scripts)
        planner: StubPlanner | LLMPlanner = StubPlanner()
        if "planner" in real:
            spec = settings.agents.planner
            planner = LLMPlanner(
                models.build("planner"),
                read_prompt(prompts_dir, spec.prompt),
                settings.guardrails.max_sub_questions,
                spec.limits.get("clarifying_questions", 3),
            )
        writer: Writer = StubWriter()
        if "writer" in real:
            spec = settings.agents.writer
            checker = ReportChecker(
                models.build("critic"),
                read_prompt(prompts_dir, settings.agents.report_check_prompt),
            )
            writer = LLMWriter(
                models.build("writer"),
                checker,
                read_prompt(prompts_dir, spec.prompt),
                audit,
                spec.limits.get("repairs", 2),
            )
        return Agents(planner, StubResearcher(budget), StubCritic(), writer)

    return make
