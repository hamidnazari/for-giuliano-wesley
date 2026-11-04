"""What a run gives its agents: budget, audit log, tool registry and the source store."""

from dataclasses import dataclass

from research_assistant.guardrails.budget import Budget
from research_assistant.observability.audit import AuditLog
from research_assistant.sources import SourceStore
from research_assistant.tools.registry import ToolRegistry


@dataclass
class AgentEnv:
    budget: Budget
    audit: AuditLog
    registry: ToolRegistry
    sources: SourceStore
