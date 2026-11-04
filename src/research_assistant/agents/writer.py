"""The Writer: turns approved cards into a cited report, then has the report checked."""

import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from research_assistant.agents.factory import AgentModel, json_instruction
from research_assistant.guardrails.untrusted import wrap_untrusted
from research_assistant.models import Board
from research_assistant.observability.audit import AuditLog
from research_assistant.report_lint import Claim, claims, lint_report
from research_assistant.reporting import (
    done_cards,
    limitations_section,
    plain_report,
    sources_section,
)

_FENCE = re.compile(r"^```(?:markdown|md)?\s*\n|\n```\s*$", re.IGNORECASE)
_H2 = re.compile(r"^##\s+(.*\S)\s*$")
_OWN_SECTIONS = {"limitations", "sources"}  # the Writer never authors these


class ClaimResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    supported: bool
    reason: str = ""


class EntailmentReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[ClaimResult]


def evidence_for(board: Board, cited: tuple[str, ...]) -> list[str]:
    """Approved note texts that cite any of these sources."""
    wanted = set(cited)
    return [n.text for c in done_cards(board) for n in c.notes if wanted & set(n.source_ids)]


class ReportChecker:
    """Final entailment pass: does each claim follow from the approved notes it cites?

    Runs on the Critic's model, a different provider from the Writer.
    """

    def __init__(self, model: AgentModel, system_prompt: str) -> None:
        self._model = model
        self._system = SystemMessage(content=system_prompt)

    def unsupported(self, board: Board, items: list[Claim]) -> list[Claim]:
        if not items:
            return []
        blocks: list[str] = []
        for i, c in enumerate(items):
            notes = "\n".join(f"  - {t}" for t in evidence_for(board, c.cited)) or "  (none)"
            blocks.append(f"Claim {i}: {c.text}\nApproved notes:\n{notes}")
        data = wrap_untrusted("\n\n".join(blocks), "report claims and approved notes")
        request = f"{data}\n\nCheck every claim. {json_instruction(EntailmentReply)}"
        reply = self._model.structured(
            [self._system, HumanMessage(content=request)], EntailmentReply
        )
        supported = {r.id for r in reply.results if r.supported}
        return [c for i, c in enumerate(items) if i not in supported]  # missing means unsupported


def _preview(text: str, limit: int = 100) -> str:
    text = text.lstrip("-*• ").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _strip_own_sections(body: str) -> str:
    """Remove any Limitations or Sources section the model wrote; code adds the real ones."""
    kept: list[str] = []
    skipping = False
    for line in body.splitlines():
        m = _H2.match(line)
        if m:
            skipping = m.group(1).lower() in _OWN_SECTIONS
        if re.match(r"^#\s", line):
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept).strip()


def _drop_lines(body: str, bad: list[Claim]) -> str:
    gone = {c.line for c in bad}
    lines = [ln for n, ln in enumerate(body.splitlines(), start=1) if n not in gone]
    out: list[str] = []
    for i, ln in enumerate(lines):  # remove headings left with nothing under them
        if re.match(r"^#{2,6}\s", ln):
            level = len(ln) - len(ln.lstrip("#"))
            nxt = next((x for x in lines[i + 1 :] if x.strip()), None)
            if nxt is None or (
                re.match(r"^#{1,6}\s", nxt) and len(nxt) - len(nxt.lstrip("#")) <= level
            ):
                continue
        out.append(ln)
    return "\n".join(out)


@dataclass
class _Parts:
    body: str
    board: Board
    stop_reason: str | None
    extra_limits: list[str]

    def assemble(self) -> str:
        head, _, rest = self.body.partition("\n")
        out = [head]
        if self.stop_reason:
            out += ["", f"> This report is partial. The run stopped early: {self.stop_reason}"]
        out += [rest] if rest.strip() else []
        out += [
            "",
            *limitations_section(self.board, self.extra_limits),
            *sources_section(self.board),
        ]
        return "\n".join(out).rstrip() + "\n"


class LLMWriter:
    def __init__(
        self,
        model: AgentModel,
        checker: ReportChecker,
        system_prompt: str,
        audit: AuditLog,
        repairs: int = 2,
    ) -> None:
        self._model = model
        self._checker = checker
        self._system = SystemMessage(content=system_prompt)
        self._audit = audit
        self._repairs = repairs

    def write(self, board: Board, question: str, stop_reason: str | None) -> str:
        cards = done_cards(board)
        if not cards:
            return plain_report(board, question, stop_reason)  # nothing approved to write about
        convo: list[BaseMessage] = [
            self._system,
            HumanMessage(content=self._request(board, question)),
        ]
        parts = _Parts("", board, stop_reason, [])
        bad: list[Claim] = []
        lint: list[str] = []
        for attempt in range(self._repairs + 1):
            reply = self._model.invoke(convo)
            parts.body = self._clean(str(reply.content), question)
            full = parts.assemble()
            lint = lint_report(full, board)
            bad = [] if lint else self._checker.unsupported(board, claims(parts.body))
            if not lint and not bad:
                self._audit.record("writer", "report_checked", {"attempts": attempt + 1})
                return full
            problems = lint or [f"unsupported by the approved notes: {c.text[:80]!r}" for c in bad]
            self._audit.record(
                "writer", "report_rejected", {"attempt": attempt + 1, "problems": problems}
            )
            convo = [
                *convo,
                AIMessage(content=parts.body),
                HumanMessage(
                    content="Fix these problems and return the whole report body again:\n"
                    + "\n".join(f"- {p}" for p in problems)
                ),
            ]
        if lint:
            self._audit.record("writer", "writer_fallback", {"reason": "lint", "problems": lint})
            return plain_report(board, question, stop_reason)
        # Only unsupported claims remain: drop them and say so.
        parts.body = _drop_lines(parts.body, bad)
        parts.extra_limits = [
            f"Removed a statement the approved notes did not support: {_preview(c.text)}"
            for c in bad
        ]
        final = parts.assemble()
        if lint_report(final, board):
            self._audit.record("writer", "writer_fallback", {"reason": "lint after drop"})
            return plain_report(board, question, stop_reason)
        self._audit.record("writer", "claims_dropped", {"count": len(bad)})
        return final

    def _request(self, board: Board, question: str) -> str:
        lines: list[str] = []
        for card in done_cards(board):
            lines.append(f"Finding group {card.id}: {card.sub_question}")
            lines += [f"- {n.text} [{', '.join(n.source_ids)}]" for n in card.notes]
            for s in card.sources:
                score = card.credibility.get(s.source_id)
                rating = f", credibility {score:.2f}" if score is not None else ""
                lines.append(f"  source [{s.source_id}]: {s.title or s.locator}{rating}")
            lines.append("")
        data = wrap_untrusted("\n".join(lines), "approved research notes")
        return (
            f"Research question: {question}\n\nApproved findings:\n{data}\n\n"
            "Write the report body now, following your rules."
        )

    @staticmethod
    def _clean(text: str, question: str) -> str:
        body = _strip_own_sections(_FENCE.sub("", text.strip()))
        return body if body.startswith("# ") else f"# {question}\n\n{body}"
