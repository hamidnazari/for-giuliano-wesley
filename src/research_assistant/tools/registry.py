"""Deny-by-default tool registry. The allowlist is enforced at build time and on every call."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from research_assistant.observability.audit import AuditLog
from research_assistant.reliability.retry import call_with_retry, call_with_timeout
from research_assistant.settings import Settings

_URL = re.compile(r"https?://", re.IGNORECASE)


class ToolNotAllowed(Exception):
    pass


class ToolArgumentRejected(Exception):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    fn: Callable[..., Any]
    allow_urls: bool = False  # only for tools that write locally and send nothing out


class Toolbox:
    """The tools one agent may call."""

    def __init__(self, agent: str, allowed: frozenset[str], registry: "ToolRegistry") -> None:
        self.agent = agent
        self.allowed = allowed
        self._registry = registry

    def call(self, tool: str, **kwargs: Any) -> Any:
        return self._registry.call(self.agent, self.allowed, tool, kwargs)


class ToolRegistry:
    def __init__(
        self,
        settings: Settings,
        audit: AuditLog,
        secrets: list[str] | None = None,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 3,
    ) -> None:
        self._settings = settings
        self._audit = audit
        self._secrets = secrets or []
        self._timeout = timeout_seconds
        self._attempts = retry_attempts
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name not in self._settings.tools.tools:
            raise ValueError(f"tool '{tool.name}' is not declared in tools.yaml")
        self._tools[tool.name] = tool

    def for_agent(self, agent: str) -> Toolbox:
        """Build-time check: the agent's configured tools must all exist and be implemented."""
        spec = self._settings.agents.by_name().get(agent)
        if spec is None:
            raise ToolNotAllowed(f"unknown agent '{agent}'")
        for name in spec.tools:
            if name not in self._tools:
                raise ToolNotAllowed(f"{agent}: tool '{name}' has no registered implementation")
        return Toolbox(agent, frozenset(spec.tools), self)

    def call(self, agent: str, allowed: frozenset[str], tool: str, kwargs: dict[str, Any]) -> Any:
        configured = set(self._settings.agents.by_name()[agent].tools)
        if tool not in allowed or tool not in configured or tool not in self._tools:
            self._audit.record(agent, "tool_blocked", {"tool": tool, "args": kwargs})
            raise ToolNotAllowed(f"{agent} may not call '{tool}'")
        problem = self._argument_problem(kwargs, self._tools[tool].allow_urls)
        if problem:
            self._audit.record(agent, "tool_rejected", {"tool": tool, "reason": problem})
            raise ToolArgumentRejected(f"{tool}: {problem}")
        self._audit.record(agent, "tool_call", {"tool": tool, "args": kwargs})
        fn = self._tools[tool].fn
        try:
            result = call_with_retry(
                lambda: call_with_timeout(lambda: fn(**kwargs), self._timeout),
                attempts=self._attempts,
            )
        except Exception as e:
            self._audit.record(agent, "tool_error", {"tool": tool, "error": repr(e)})
            raise
        self._audit.record(agent, "tool_result", {"tool": tool})
        return result

    def _argument_problem(self, kwargs: dict[str, Any], allow_urls: bool) -> str | None:
        """No URL may be passed in; no secret may leave through an argument."""
        for key, value in kwargs.items():
            if not isinstance(value, str):
                continue
            if not allow_urls and _URL.search(value):
                return f"argument '{key}' contains a URL; tools take IDs, not URLs"
            if any(s in value for s in self._secrets):
                return f"argument '{key}' contains a secret"
        return None
