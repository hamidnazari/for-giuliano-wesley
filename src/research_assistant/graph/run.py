# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""One research run, start to finish: wiring, execution, and clean stops."""

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langgraph.errors import GraphRecursionError

from research_assistant.agents.protocols import Agents
from research_assistant.board.kanzo import KanzoSink, SafeSink
from research_assistant.board.sink import BoardSink, JsonSnapshotSink
from research_assistant.graph.workflow import RunContext, build_graph, write_report
from research_assistant.guardrails.budget import Budget, BudgetExceeded
from research_assistant.interaction import UserIO
from research_assistant.models import Board
from research_assistant.observability.audit import AuditLog
from research_assistant.observability.run import new_run_id, project_dir
from research_assistant.settings import Settings
from research_assistant.tools.registry import Tool, ToolRegistry
from research_assistant.tools.render import make_render_report

Status = Literal["completed", "stopped", "declined"]


@dataclass
class RunResult:
    run_id: str
    status: Status
    board: Board
    report_path: str | None
    stop_reason: str | None
    project_dir: Path


def run_research(
    question: str,
    settings: Settings,
    data_root: Path,
    io: UserIO,
    make_agents: Any,
    *,
    kanzo: bool = False,
    kanzo_home: Path | None = None,
    budget: Budget | None = None,
    recursion_limit: int | None = None,
) -> RunResult:
    """Run the workflow. make_agents(budget) returns the Agents to use.

    A tripped cap or step limit ends the run cleanly: the partial board is saved and the
    report notes the gaps. Rerunning starts fresh.
    """
    run_id = new_run_id()
    pdir = project_dir(data_root, run_id)
    audit = AuditLog(pdir / "audit.jsonl", run_id)
    budget = budget or Budget(settings.guardrails)
    registry = ToolRegistry(settings, audit)
    registry.register(Tool("render_report", make_render_report(pdir), allow_urls=True))
    agents: Agents = make_agents(budget)
    guard = settings.guardrails
    limit = recursion_limit or 20 + 8 * (guard.max_critic_rounds + guard.max_card_attempts)

    with ExitStack() as stack:
        sinks: list[BoardSink] = [JsonSnapshotSink(pdir / "board.json")]
        if kanzo:
            sinks.append(SafeSink(stack.enter_context(KanzoSink(run_id, home=kanzo_home)), audit))
        ctx = RunContext(
            run_id, settings, budget, audit, agents, io, registry.for_agent("writer"), sinks
        )
        graph = build_graph(ctx)
        audit.record("run", "run_started", {"question": question})

        state: dict[str, Any] = {
            "question": question,
            "plan": None,
            "approved": False,
            "board": Board(run_id=run_id),
            "report_path": None,
        }
        stop_reason: str | None = None
        try:
            for latest in graph.stream(
                state, config={"recursion_limit": limit}, stream_mode="values"
            ):
                state = latest
        except BudgetExceeded as e:
            stop_reason = str(e)
        except GraphRecursionError:
            stop_reason = f"step limit of {limit} reached"

        board: Board = state["board"]
        if stop_reason:
            audit.record("run", "run_stopped", {"reason": stop_reason})
            path = write_report(ctx, board, question, stop_reason)
            status: Status = "stopped"
        elif state["plan"] is not None and not state["approved"]:
            audit.record("run", "run_declined", {})
            path, status = None, "declined"
        else:
            path, status = state["report_path"], "completed"
        audit.record("run", "run_finished", {"status": status})
        return RunResult(run_id, status, board, path, stop_reason, pdir)
