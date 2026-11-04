"""What the graph needs from each agent. Stubs and real agents both satisfy these."""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from research_assistant.interaction import UserIO
from research_assistant.models import Board, Card, Note, Plan, Source, Verdict


class Work(BaseModel):
    """What a Researcher hands in for a card."""

    model_config = ConfigDict(extra="forbid")

    notes: list[Note]
    sources: list[Source]
    credibility: dict[str, float] = {}


class Planner(Protocol):
    def plan(self, question: str, io: UserIO) -> Plan: ...


class Researcher(Protocol):
    def research(self, card: Card) -> Work: ...


class Critic(Protocol):
    def review(self, card: Card, round_number: int) -> Verdict: ...


class Writer(Protocol):
    def write(self, board: Board, question: str, stop_reason: str | None) -> str:
        """Return the report as markdown."""
        ...


@dataclass
class Agents:
    planner: Planner
    researcher: Researcher
    critic: Critic
    writer: Writer
