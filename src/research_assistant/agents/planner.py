"""The Planner: asks a few clarifying questions, then proposes sub-questions."""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, create_model

from research_assistant.agents.factory import AgentModel, json_instruction
from research_assistant.interaction import UserIO
from research_assistant.models import Plan


class Clarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[str] = Field(default=[], description="Empty when the question is clear.")


class LLMPlanner:
    def __init__(
        self,
        model: AgentModel,
        system_prompt: str,
        max_sub_questions: int,
        max_clarifying: int = 3,
    ) -> None:
        self._model = model
        self._system = SystemMessage(content=system_prompt)
        self._max_clarifying = max_clarifying
        # The cap is part of the schema, so an over-long plan is repaired by the retry loop.
        self._draft: type[BaseModel] = create_model(
            "PlanDraft",
            __config__=ConfigDict(extra="forbid"),
            question=(str, Field(min_length=1, description="The question, restated clearly.")),
            sub_questions=(list[str], Field(min_length=1, max_length=max_sub_questions)),
        )
        self._max_sub = max_sub_questions

    def plan(self, question: str, io: UserIO) -> Plan:
        answers = self._clarify(question, io) if io.interactive else []
        context = f"Research question: {question}"
        if answers:
            context += "\n\nClarifications from the user:\n" + "\n".join(
                f"Q: {q}\nA: {a}" for q, a in answers
            )
        request = (
            f"{context}\n\nWrite the research plan with at most {self._max_sub} sub-questions. "
            + json_instruction(self._draft)
        )
        draft = self._model.structured([self._system, HumanMessage(content=request)], self._draft)
        return Plan.model_validate(draft.model_dump())

    def _clarify(self, question: str, io: UserIO) -> list[tuple[str, str]]:
        request = (
            f"Research question: {question}\n\nDo you need to ask the user anything before "
            f"planning? Ask at most {self._max_clarifying} questions, or none. "
            + json_instruction(Clarification)
        )
        found = self._model.structured([self._system, HumanMessage(content=request)], Clarification)
        answers: list[tuple[str, str]] = []
        for q in found.questions[: self._max_clarifying]:
            a = io.ask(q)
            if a:
                answers.append((q, a))
        return answers
