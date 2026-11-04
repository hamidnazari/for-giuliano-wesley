"""Model factory: builds each agent's model from config, real (Kilo gateway) or scripted."""

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from research_assistant.guardrails.budget import Budget
from research_assistant.observability.audit import AuditLog
from research_assistant.reliability.retry import call_with_retry, call_with_timeout
from research_assistant.settings import Settings

KILO_BASE_URL = "https://ai-gateway.kilo.ai/api/v1"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# model id -> (input price, output price) per token
Prices = Mapping[str, tuple[float, float]]


class MalformedOutput(Exception):
    """The model kept returning output that does not match the schema."""


def scripted(responses: Sequence[str]) -> BaseChatModel:
    """A fake model that returns the given responses in order. For tests and CI."""
    return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in responses]))


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content)


class AgentModel:
    """A model bound to one agent: enforces budget, retries, logs usage and cost."""

    def __init__(
        self,
        agent: str,
        model_id: str,
        model: BaseChatModel,
        budget: Budget,
        audit: AuditLog,
        prices: Prices | None = None,
        timeout_seconds: float = 120.0,
        attempts: int = 3,
    ) -> None:
        self.agent = agent
        self.model_id = model_id
        self._model = model
        self._budget = budget
        self._audit = audit
        self._prices = prices or {}
        self._timeout = timeout_seconds
        self._attempts = attempts

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        self._budget.before_llm_call()
        reply = call_with_retry(
            lambda: call_with_timeout(lambda: self._model.invoke(list(messages)), self._timeout),
            attempts=self._attempts,
        )
        self._record(messages, reply)
        return reply

    def structured[T: BaseModel](
        self, messages: Sequence[BaseMessage], schema: type[T], max_repairs: int = 2
    ) -> T:
        """Get JSON matching schema; on malformed output, show the model its error and retry."""
        convo = list(messages)
        last_error = ""
        for _ in range(max_repairs + 1):
            reply = self.invoke(convo)
            raw = _FENCE.sub("", _text(reply.content).strip())
            try:
                return schema.model_validate_json(raw)
            except ValidationError as e:
                last_error = str(e)
                self._audit.record(self.agent, "malformed_output", {"error": last_error[:500]})
                convo = [
                    *convo,
                    reply,
                    HumanMessage(
                        content="Your reply was not valid JSON for the required schema. "
                        f"Fix it and reply with JSON only. Error: {last_error[:500]}"
                    ),
                ]
        raise MalformedOutput(f"{self.agent}: no valid output after retries: {last_error[:200]}")

    def _record(self, messages: Sequence[BaseMessage], reply: AIMessage) -> None:
        usage = reply.usage_metadata
        if usage:
            tokens_in, tokens_out = usage["input_tokens"], usage["output_tokens"]
            estimated = False
        else:
            tokens_in = sum(_estimate_tokens(_text(m.content)) for m in messages)
            tokens_out = _estimate_tokens(_text(reply.content))
            estimated = True
        price_in, price_out = self._prices.get(self.model_id, (0.0, 0.0))
        cost = tokens_in * price_in + tokens_out * price_out
        self._budget.record_llm_call(self.agent, tokens_in, tokens_out, cost)
        self._audit.record(
            self.agent,
            "llm_call",
            {
                "model": self.model_id,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "estimated": estimated,
                "cost": cost,
            },
        )


class ModelFactory:
    def __init__(
        self,
        settings: Settings,
        budget: Budget,
        audit: AuditLog,
        scripts: Mapping[str, Sequence[str]] | None = None,
        prices: Prices | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._budget = budget
        self._audit = audit
        self._scripts = scripts or {}
        self._prices = prices
        self._env = os.environ if env is None else env

    def build(self, agent: str) -> AgentModel:
        spec = self._settings.agents.by_name().get(agent)
        if spec is None:
            raise ValueError(f"unknown agent '{agent}'")
        if agent in self._scripts:
            model = scripted(self._scripts[agent])
        else:
            key = self._env.get("KILO_API_KEY")
            if not key:
                raise RuntimeError("KILO_API_KEY is not set")
            # Our own retry and timeout wrap every call, so the client does not retry.
            model = ChatOpenAI(
                model=spec.model,
                api_key=SecretStr(key),
                base_url=KILO_BASE_URL,
                max_retries=0,
            )
        return AgentModel(agent, spec.model, model, self._budget, self._audit, self._prices)
