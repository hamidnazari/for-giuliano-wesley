"""Board operations. Every change to a card is one of these, applied by the reducer.

Each op carries an op_id. Callers derive it deterministically from context (for example
"submit:c1:2" for the second submission of card c1), so replaying an op is a no-op.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.models import Note, Source, Verdict


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    op_id: str = Field(min_length=1)


class CreateCard(_Op):
    type: Literal["create"] = "create"
    card_id: str = Field(min_length=1)
    sub_question: str = Field(min_length=1)


class Claim(_Op):
    """A Researcher takes a Backlog card."""

    type: Literal["claim"] = "claim"
    card_id: str


class SubmitWork(_Op):
    """A Researcher hands in notes and sources; the card goes to review."""

    type: Literal["submit"] = "submit"
    card_id: str
    notes: list[Note]
    sources: list[Source]
    credibility: dict[str, float] = {}


class Review(_Op):
    """The Critic's verdict on a card in review."""

    type: Literal["review"] = "review"
    card_id: str
    verdict: Verdict


class Block(_Op):
    """A failed card is parked with its reason instead of being dropped."""

    type: Literal["block"] = "block"
    card_id: str
    reason: str = Field(min_length=1)


class Reject(_Op):
    """A card gives up, for example after its attempt limit."""

    type: Literal["reject"] = "reject"
    card_id: str
    reason: str = Field(min_length=1)


Op = Annotated[
    CreateCard | Claim | SubmitWork | Review | Block | Reject,
    Field(discriminator="type"),
]
