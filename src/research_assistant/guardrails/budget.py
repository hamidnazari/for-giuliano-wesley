"""Run budget: hard caps, cost tracking, and per-card limits."""

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from research_assistant.settings import GuardrailsConfig


class BudgetExceeded(Exception):
    """A run-level cap tripped; the run ends cleanly with its partial board."""

    def __init__(self, limit: str, used: float, cap: float) -> None:
        self.limit = limit
        super().__init__(f"budget exceeded: {limit} {used:g} > {cap:g}")


class CardLimitExceeded(Exception):
    """A per-card limit tripped; the card moves to Rejected or escalates, the run continues."""

    def __init__(self, card_id: str, limit: str, cap: int) -> None:
        self.card_id = card_id
        self.limit = limit
        super().__init__(f"card {card_id}: {limit} limit of {cap} reached")


@dataclass
class AgentUsage:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class Budget:
    def __init__(self, caps: GuardrailsConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self._caps = caps
        self._clock = clock
        self._start = clock()
        self._lock = threading.Lock()
        self.searches = 0
        self.fetches = 0
        self.usage: dict[str, AgentUsage] = defaultdict(AgentUsage)
        self._attempts: dict[str, int] = defaultdict(int)
        self._rounds: dict[str, int] = defaultdict(int)

    # run-level
    @property
    def llm_calls(self) -> int:
        return sum(u.llm_calls for u in self.usage.values())

    @property
    def tokens(self) -> int:
        return sum(u.input_tokens + u.output_tokens for u in self.usage.values())

    @property
    def total_cost(self) -> float:
        return sum(u.cost for u in self.usage.values())

    def check_wall_clock(self) -> None:
        elapsed = self._clock() - self._start
        if elapsed > self._caps.max_wall_clock_seconds:
            raise BudgetExceeded("wall_clock_seconds", elapsed, self._caps.max_wall_clock_seconds)

    def before_llm_call(self) -> None:
        """Call before each model call; raises if the next call would exceed a cap."""
        self.check_wall_clock()
        with self._lock:
            if self.llm_calls + 1 > self._caps.max_llm_calls:
                raise BudgetExceeded("llm_calls", self.llm_calls + 1, self._caps.max_llm_calls)
            if self.tokens >= self._caps.max_tokens:
                raise BudgetExceeded("tokens", self.tokens, self._caps.max_tokens)

    def record_llm_call(
        self, agent: str, input_tokens: int, output_tokens: int, cost: float = 0.0
    ) -> None:
        with self._lock:
            u = self.usage[agent]
            u.llm_calls += 1
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens
            u.cost += cost

    def before_search(self) -> None:
        self.check_wall_clock()
        with self._lock:
            if self.searches + 1 > self._caps.max_searches:
                raise BudgetExceeded("searches", self.searches + 1, self._caps.max_searches)
            self.searches += 1

    def before_fetch(self) -> None:
        self.check_wall_clock()
        with self._lock:
            if self.fetches + 1 > self._caps.max_page_fetches:
                raise BudgetExceeded("page_fetches", self.fetches + 1, self._caps.max_page_fetches)
            self.fetches += 1

    # per-card
    def start_attempt(self, card_id: str) -> int:
        with self._lock:
            if self._attempts[card_id] >= self._caps.max_card_attempts:
                raise CardLimitExceeded(card_id, "attempts", self._caps.max_card_attempts)
            self._attempts[card_id] += 1
            return self._attempts[card_id]

    def start_critic_round(self, card_id: str) -> int:
        with self._lock:
            if self._rounds[card_id] >= self._caps.max_critic_rounds:
                raise CardLimitExceeded(card_id, "critic_rounds", self._caps.max_critic_rounds)
            self._rounds[card_id] += 1
            return self._rounds[card_id]
