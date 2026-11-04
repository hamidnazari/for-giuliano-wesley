# pyright: reportUnknownMemberType=false, reportArgumentType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false
# LangGraph's node typing cannot express Send payload nodes; the rest of the file stays strict.
"""The LangGraph workflow. The board channel's reducer is the only place cards change."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from research_assistant.agents.protocols import Agents
from research_assistant.board.ops import Block, Claim, CreateCard, Op, Reject, Review, SubmitWork
from research_assistant.board.reducer import apply
from research_assistant.board.sink import BoardSink
from research_assistant.guardrails.budget import Budget, BudgetExceeded, CardLimitExceeded
from research_assistant.interaction import UserIO
from research_assistant.models import Board, Card, CardStatus, Plan, Verdict, VerdictKind
from research_assistant.observability.audit import AuditLog
from research_assistant.settings import Settings
from research_assistant.tools.registry import Toolbox


@dataclass
class RunContext:
    run_id: str
    settings: Settings
    budget: Budget
    audit: AuditLog
    agents: Agents
    io: UserIO
    writer_tools: Toolbox
    sinks: Sequence[BoardSink]


def write_report(ctx: RunContext, board: Board, question: str, stop_reason: str | None) -> str:
    """Ask the Writer for markdown and save it with the render_report tool."""
    markdown = ctx.agents.writer.write(board, question, stop_reason)
    path = ctx.writer_tools.call("render_report", markdown=markdown)
    ctx.audit.record("writer", "report_written", {"path": path, "partial": bool(stop_reason)})
    return str(path)


def build_graph(ctx: RunContext) -> CompiledStateGraph[Any, Any, Any, Any]:
    cap = ctx.settings.guardrails.max_sub_questions

    def reduce_board(left: Board, right: list[Op]) -> Board:
        """The single writer. Applies a batch of ops, then records and mirrors it."""
        new = apply(left, right, cap)
        fresh = [op for op in right if op.op_id not in left.applied_ops]
        for op in fresh:
            ctx.audit.record("board", "board_op", op.model_dump(mode="json"))
        for sink in ctx.sinks:
            sink.sync(new, fresh)
        return new

    class RunState(TypedDict):
        question: str
        plan: Plan | None
        approved: bool
        board: Annotated[Board, reduce_board]
        report_path: str | None

    def plan_node(state: RunState) -> dict[str, Any]:
        plan = ctx.agents.planner.plan(state["question"], ctx.io)
        plan.check_cap(cap)
        ctx.audit.record("planner", "plan_created", plan.model_dump())
        return {"plan": plan}

    def approve_node(state: RunState) -> dict[str, Any]:
        assert state["plan"] is not None
        approved = ctx.io.approve_plan(state["plan"])
        ctx.audit.record("user", "plan_approval", {"approved": approved})
        return {"approved": approved}

    def after_approval(state: RunState) -> str:
        return "init_cards" if state["approved"] else END

    def init_cards(state: RunState) -> dict[str, Any]:
        assert state["plan"] is not None
        ops: list[Op] = [
            CreateCard(op_id=f"create:c{i}", card_id=f"c{i}", sub_question=q)
            for i, q in enumerate(state["plan"].sub_questions, start=1)
        ]
        return {"board": ops}

    def dispatch(state: RunState) -> dict[str, Any]:
        ctx.budget.check_wall_clock()
        return {}

    def route(state: RunState) -> list[Send] | str:
        cards = list(state["board"].cards.values())
        count = len(cards)
        todo = [c for c in cards if c.status in (CardStatus.BACKLOG, CardStatus.IN_PROGRESS)]
        if todo:
            return [Send("research", {"card": c}) for c in todo]
        in_review = [c for c in cards if c.status is CardStatus.IN_REVIEW]
        if in_review:
            return [Send("review", {"card": c, "card_count": count}) for c in in_review]
        return "write"

    def research(arg: dict[str, Any]) -> dict[str, Any]:
        card: Card = arg["card"]
        try:
            attempt = ctx.budget.start_attempt(card.id)
        except CardLimitExceeded as e:
            return {
                "board": [
                    Reject(op_id=f"reject:{card.id}:attempts", card_id=card.id, reason=str(e))
                ]
            }
        ops: list[Op] = []
        if card.status is CardStatus.BACKLOG:
            ops.append(Claim(op_id=f"claim:{card.id}:{attempt}", card_id=card.id))
        try:
            work = ctx.agents.researcher.research(card)
        except BudgetExceeded:
            raise
        except Exception as e:
            ctx.audit.record("researcher", "research_failed", {"card": card.id, "error": repr(e)})
            ops.append(
                Block(
                    op_id=f"block:{card.id}:{attempt}",
                    card_id=card.id,
                    reason=f"research failed: {e!r}",
                )
            )
            return {"board": ops}
        ops.append(
            SubmitWork(
                op_id=f"submit:{card.id}:{attempt}",
                card_id=card.id,
                notes=work.notes,
                sources=work.sources,
                credibility=work.credibility,
            )
        )
        return {"board": ops}

    def review(arg: dict[str, Any]) -> dict[str, Any]:
        card: Card = arg["card"]
        try:
            round_number = ctx.budget.start_critic_round(card.id)
        except CardLimitExceeded:
            open_items = "; ".join(card.critic_feedback) or "none recorded"
            reason = f"needs your decision: critic rounds exhausted. Open objections: {open_items}"
            return {
                "board": [Block(op_id=f"block:{card.id}:rounds", card_id=card.id, reason=reason)]
            }
        verdict = ctx.agents.critic.review(card, round_number)
        if verdict.kind is VerdictKind.SPLIT and arg["card_count"] + len(verdict.split_into) > cap:
            verdict = Verdict(
                card_id=card.id,
                kind=VerdictKind.REJECT,
                rubric_version=verdict.rubric_version,
                objections=[*verdict.objections, "split would exceed the sub-question cap"],
            )
        return {
            "board": [
                Review(op_id=f"review:{card.id}:{round_number}", card_id=card.id, verdict=verdict)
            ]
        }

    def write(state: RunState) -> dict[str, Any]:
        return {"report_path": write_report(ctx, state["board"], state["question"], None)}

    g = StateGraph(RunState)
    g.add_node("plan", plan_node)
    g.add_node("approve", approve_node)
    g.add_node("init_cards", init_cards)
    g.add_node("dispatch", dispatch)
    g.add_node("research", research)
    g.add_node("review", review)
    g.add_node("write", write)
    g.add_edge(START, "plan")
    g.add_edge("plan", "approve")
    g.add_conditional_edges("approve", after_approval, ["init_cards", END])
    g.add_edge("init_cards", "dispatch")
    g.add_conditional_edges("dispatch", route, ["research", "review", "write"])
    g.add_edge("research", "dispatch")
    g.add_edge("review", "dispatch")
    g.add_edge("write", END)
    return g.compile()
