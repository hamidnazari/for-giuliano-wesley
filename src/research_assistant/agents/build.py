"""Pick real or stub agents for a run. Agents not yet built are always stubs."""

from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path

from research_assistant.agents.critic import LLMCritic
from research_assistant.agents.env import AgentEnv
from research_assistant.agents.factory import ModelFactory
from research_assistant.agents.planner import LLMPlanner
from research_assistant.agents.prompts import read_prompt
from research_assistant.agents.protocols import Agents, Critic, Researcher, Writer
from research_assistant.agents.researcher import LLMResearcher
from research_assistant.agents.stubs import StubCritic, StubPlanner, StubResearcher, StubWriter
from research_assistant.agents.writer import LLMWriter, ReportChecker
from research_assistant.documents.index import DocIndex
from research_assistant.settings import Settings
from research_assistant.tools.docs import make_docs_retrieve
from research_assistant.tools.passages import PassageCache, make_fetch_cited_passage
from research_assistant.tools.registry import Tool
from research_assistant.tools.search import ExtractBackend, SearchBackend, make_web_search

REAL_AGENTS = ("planner", "researcher", "critic", "writer")  # grows as real agents are built


def agents_builder(
    settings: Settings,
    prompts_dir: Path,
    real: Collection[str] = (),
    scripts: Mapping[str, Sequence[str]] | None = None,
    search_backend: SearchBackend | None = None,
    docs_index: DocIndex | None = None,
    page_backend: ExtractBackend | None = None,
) -> Callable[[AgentEnv], Agents]:
    """Return make_agents(env). scripts feed fake models in tests."""
    unknown = set(real) - set(REAL_AGENTS)
    if unknown:
        raise ValueError(f"no real implementation yet for: {sorted(unknown)}")
    if "critic" in real and page_backend is None:
        raise ValueError("the real critic needs a page backend to fetch cited pages")
    if "researcher" in real and search_backend is None:
        raise ValueError("the real researcher needs a search backend")

    def make(env: AgentEnv) -> Agents:
        budget, audit = env.budget, env.audit
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
        researcher: Researcher = StubResearcher(budget)
        if "researcher" in real:
            assert search_backend is not None
            spec = settings.agents.researcher
            env.registry.register(
                Tool("web_search", make_web_search(search_backend, budget, settings.tools.search))
            )
            ing = settings.tools.ingestion
            env.registry.register(
                Tool("docs_retrieve", make_docs_retrieve(docs_index, ing.top_k, ing.min_score))
            )
            researcher = LLMResearcher(
                models.build("researcher"),
                read_prompt(prompts_dir, spec.prompt),
                env.registry.for_agent("researcher"),
                env.sources,
                audit,
                spec.limits.get("searches", 6),
                use_documents=docs_index is not None,
            )
        critic: Critic = StubCritic()
        if "critic" in real:
            assert page_backend is not None
            fetch = make_fetch_cited_passage(env.sources, page_backend, PassageCache(), budget)
            env.registry.register(Tool("fetch_cited_passage", fetch))
            prompt = read_prompt(prompts_dir, settings.agents.critic.prompt)
            rubric = read_prompt(prompts_dir, settings.agents.critic_rubric)
            critic = LLMCritic(
                models.build("critic"),
                f"{prompt}\n\n{rubric}",
                env.registry.for_agent("critic"),
                audit,
                settings.agents.rubric_version,
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
        return Agents(planner, researcher, critic, writer)

    return make
