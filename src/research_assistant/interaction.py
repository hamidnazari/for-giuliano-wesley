"""How the run talks to the user: clarifying questions and the plan approval gate."""

from collections.abc import Sequence
from typing import Protocol

from research_assistant.models import Plan


class UserIO(Protocol):
    @property
    def interactive(self) -> bool:
        """False when nobody can answer questions, so agents should not ask any."""
        ...

    def ask(self, question: str) -> str: ...

    def approve_plan(self, plan: Plan) -> bool: ...


def format_plan(plan: Plan) -> str:
    lines = [f"Question: {plan.question}", "Sub-questions:"]
    lines += [f"  {i}. {q}" for i, q in enumerate(plan.sub_questions, start=1)]
    return "\n".join(lines)


class ConsoleIO:
    """Interactive. Closing stdin or pressing Ctrl-C at a gate declines; the run exits cleanly."""

    interactive = True

    def ask(self, question: str) -> str:
        try:
            return input(f"{question}\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    def approve_plan(self, plan: Plan) -> bool:
        print(format_plan(plan))
        try:
            return input("Approve this plan and start research? [y/N] ").strip().lower() in {
                "y",
                "yes",
            }
        except (EOFError, KeyboardInterrupt):
            return False


class AutoApproveIO:
    """Non-interactive, for CI and evals: no clarifying answers, plan always approved."""

    interactive = False

    def ask(self, question: str) -> str:
        return ""

    def approve_plan(self, plan: Plan) -> bool:
        return True


class ScriptedIO:
    """Answers from lists, for tests."""

    def __init__(
        self, answers: Sequence[str] = (), approve: bool = True, interactive: bool = True
    ) -> None:
        self.interactive = interactive
        self._answers = list(answers)
        self._approve = approve
        self.asked: list[str] = []

    def ask(self, question: str) -> str:
        self.asked.append(question)
        return self._answers.pop(0) if self._answers else ""

    def approve_plan(self, plan: Plan) -> bool:
        return self._approve
