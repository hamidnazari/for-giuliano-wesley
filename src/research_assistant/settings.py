"""Config schemas, loader and startup validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

MODEL_ID = r"^[a-z0-9][a-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.:-]*$"  # provider/model-name
READ_ONLY_AGENTS = ("planner", "researcher", "critic")
CRITIC_TOOLS = {"fetch_cited_passage"}


class ConfigError(Exception):
    """Raised with every problem found, so one run shows them all."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("invalid config:\n" + "\n".join(f"  - {p}" for p in problems))


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentSpec(Strict):
    model: str = Field(pattern=MODEL_ID)
    prompt: str = Field(min_length=1)  # file name under config/prompts
    tools: list[str] = []
    limits: dict[str, int] = {}


class AgentsConfig(Strict):
    planner: AgentSpec
    researcher: AgentSpec
    critic: AgentSpec
    writer: AgentSpec
    critic_rubric: str = "critic_rubric.md"
    report_check_prompt: str = "report_check.md"  # final entailment pass over the report
    rubric_version: str = Field(min_length=1)

    def by_name(self) -> dict[str, AgentSpec]:
        return {n: getattr(self, n) for n in ("planner", "researcher", "critic", "writer")}


class ToolSpec(Strict):
    source: Literal["native", "mcp"]
    risk: Literal["read", "write"]


class SearchConfig(Strict):
    search_depth: Literal["basic", "advanced"] = "basic"
    max_results: int = Field(default=5, gt=0)


class IngestionConfig(Strict):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    embedding_model: str = Field(min_length=1)
    top_k: int = Field(default=4, gt=0)
    min_score: float = Field(default=0.0, ge=0, le=1)


class ToolsConfig(Strict):
    tools: dict[str, ToolSpec]
    search: SearchConfig = SearchConfig()
    ingestion: IngestionConfig


class GuardrailsConfig(Strict):
    max_llm_calls: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    max_searches: int = Field(gt=0)
    max_page_fetches: int = Field(default=100, gt=0)  # full-page fetches by the Critic
    max_wall_clock_seconds: int = Field(gt=0)
    max_critic_rounds: int = Field(gt=0)
    max_card_attempts: int = Field(gt=0)
    max_sub_questions: int = Field(gt=0)


class ModelPrice(Strict):
    input: float = Field(ge=0)  # USD per million tokens
    output: float = Field(ge=0)


class PricesConfig(Strict):
    models: dict[str, ModelPrice]


class Settings(Strict):
    agents: AgentsConfig
    tools: ToolsConfig
    guardrails: GuardrailsConfig
    prices: PricesConfig

    def token_prices(self) -> dict[str, tuple[float, float]]:
        """Model ID -> (input, output) price per token."""
        return {m: (p.input / 1e6, p.output / 1e6) for m, p in self.prices.models.items()}


def _load_yaml(path: Path, problems: list[str]) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except FileNotFoundError:
        problems.append(f"{path.name}: file not found")
    except yaml.YAMLError as e:
        problems.append(f"{path.name}: invalid YAML ({e})")
    return None


def _parse[T: BaseModel](model: type[T], path: Path, problems: list[str]) -> T | None:
    data = _load_yaml(path, problems)
    if data is None:
        return None
    try:
        return model.model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            problems.append(f"{path.name}: {loc}: {err['msg']}")
        return None


def _provider(model_id: str) -> str:
    return model_id.split("/", 1)[0]


def _cross_check(s: Settings, prompts: Path) -> list[str]:
    problems: list[str] = []
    known = s.tools.tools
    agents = s.agents.by_name()
    for name, spec in agents.items():
        for tool in spec.tools:
            if tool not in known:
                problems.append(f"agents.yaml: {name}: unknown tool '{tool}'")
            elif name in READ_ONLY_AGENTS and known[tool].risk == "write":
                problems.append(f"agents.yaml: {name}: write-capable tool '{tool}' not allowed")
        if not (prompts / spec.prompt).is_file():
            problems.append(f"agents.yaml: {name}: prompt file '{spec.prompt}' not found")
    if agents["planner"].tools:
        problems.append("agents.yaml: planner: must have no tools")
    extra = set(agents["critic"].tools) - CRITIC_TOOLS
    if extra:
        problems.append(
            f"agents.yaml: critic: only {sorted(CRITIC_TOOLS)} allowed, got {sorted(extra)}"
        )
    if _provider(agents["critic"].model) == _provider(agents["researcher"].model):
        problems.append(
            "agents.yaml: critic must use a different model provider from researcher "
            "(the same provider shares blind spots)"
        )
    for name, spec in agents.items():
        if spec.model not in s.prices.models:
            problems.append(f"prices.yaml: no price for {name}'s model '{spec.model}'")
    if not (prompts / s.agents.critic_rubric).is_file():
        problems.append(f"agents.yaml: critic rubric '{s.agents.critic_rubric}' not found")
    first = agents["researcher"].limits.get("searches")
    revise = agents["researcher"].limits.get("revision_searches", 0)
    per_card = first + revise if first else None  # one revision round for every card
    needed = s.guardrails.max_sub_questions * per_card if per_card else 0
    if needed > s.guardrails.max_searches:
        problems.append(
            f"guardrails.yaml: max_searches {s.guardrails.max_searches} cannot cover "
            f"max_sub_questions {s.guardrails.max_sub_questions} x researcher searches "
            f"({first} + {revise} for one revision) = {needed}"
        )
    if not (prompts / s.agents.report_check_prompt).is_file():
        problems.append(
            f"agents.yaml: report check prompt '{s.agents.report_check_prompt}' not found"
        )
    ing = s.tools.ingestion
    if ing.chunk_overlap >= ing.chunk_size:
        problems.append("tools.yaml: ingestion: chunk_overlap must be smaller than chunk_size")
    return problems


def load_settings(config_dir: Path) -> Settings:
    """Load and validate all config; raise ConfigError listing every problem."""
    problems: list[str] = []
    agents = _parse(AgentsConfig, config_dir / "agents.yaml", problems)
    tools = _parse(ToolsConfig, config_dir / "tools.yaml", problems)
    guardrails = _parse(GuardrailsConfig, config_dir / "guardrails.yaml", problems)
    prices = _parse(PricesConfig, config_dir / "prices.yaml", problems)
    if agents is None or tools is None or guardrails is None or prices is None:
        raise ConfigError(problems)
    settings = Settings(agents=agents, tools=tools, guardrails=guardrails, prices=prices)
    problems = _cross_check(settings, config_dir / "prompts")
    if problems:
        raise ConfigError(problems)
    return settings
