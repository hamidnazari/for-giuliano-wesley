"""The single writer: a pure function that applies ops to a board."""

from collections.abc import Sequence
from typing import Any

from research_assistant.board.ops import (
    Block,
    Claim,
    CreateCard,
    Op,
    Reject,
    Review,
    SubmitWork,
)
from research_assistant.models import Board, Card, CardStatus, Verdict, VerdictKind

S = CardStatus

# Allowed moves. Done and Rejected are final.
TRANSITIONS: dict[CardStatus, frozenset[CardStatus]] = {
    S.BACKLOG: frozenset({S.IN_PROGRESS, S.BLOCKED, S.REJECTED}),
    S.IN_PROGRESS: frozenset({S.IN_REVIEW, S.BLOCKED, S.REJECTED}),
    S.IN_REVIEW: frozenset({S.DONE, S.IN_PROGRESS, S.BLOCKED, S.REJECTED}),
    S.BLOCKED: frozenset({S.REJECTED}),
    S.DONE: frozenset(),
    S.REJECTED: frozenset(),
}


class BoardError(Exception):
    pass


class InvalidTransition(BoardError):
    pass


class UnknownCard(BoardError):
    pass


class DuplicateCard(BoardError):
    pass


class CardCapExceeded(BoardError):
    """Creating or splitting cards would pass the sub-question cap."""


def _get(board: Board, card_id: str) -> Card:
    try:
        return board.cards[card_id]
    except KeyError:
        raise UnknownCard(card_id) from None


def _update(card: Card, status: CardStatus, **changes: Any) -> Card:
    if status not in TRANSITIONS[card.status]:
        raise InvalidTransition(f"{card.id}: {card.status.value} -> {status.value}")
    # Re-validate so Card's rules (Done needs sources, Blocked needs a reason) always hold.
    return Card.model_validate({**card.model_dump(), "status": status, **changes})


def _create(
    board: Board, card_id: str, sub_question: str, parent_id: str | None, cap: int | None
) -> None:
    if card_id in board.cards:
        raise DuplicateCard(card_id)
    if cap is not None and len(board.cards) + 1 > cap:
        raise CardCapExceeded(f"cannot create {card_id}: cap of {cap} cards reached")
    board.cards[card_id] = Card(id=card_id, sub_question=sub_question, parent_id=parent_id)


def _apply_one(board: Board, op: Op, cap: int | None) -> None:
    match op:
        case CreateCard():
            _create(board, op.card_id, op.sub_question, None, cap)
        case Claim():
            c = _get(board, op.card_id)
            board.cards[c.id] = _update(c, S.IN_PROGRESS, attempts=c.attempts + 1)
        case SubmitWork():
            c = _get(board, op.card_id)
            board.cards[c.id] = _update(
                c,
                S.IN_REVIEW,
                notes=[n.model_dump() for n in op.notes],
                sources=[s.model_dump() for s in op.sources],
                credibility=op.credibility,
            )
        case Block():
            c = _get(board, op.card_id)
            board.cards[c.id] = _update(c, S.BLOCKED, reason=op.reason)
        case Reject():
            c = _get(board, op.card_id)
            board.cards[c.id] = _update(c, S.REJECTED, reason=op.reason)
        case Review():
            _review(board, op, cap)


def _prune(card: Card, v: Verdict, credibility: dict[str, float]) -> dict[str, Any]:
    """The card's fields after removing the notes the Critic flagged.

    Surviving notes were each judged supported, so they are frozen: verified_notes counts them.
    Sources no surviving note cites are dropped with them.
    """
    gone = set(v.dropped_notes)
    if any(i > len(card.notes) for i in gone):
        raise BoardError(f"{card.id}: verdict drops a note that does not exist")
    survivors = [n for i, n in enumerate(card.notes, start=1) if i not in gone]
    cited = {sid for n in survivors for sid in n.source_ids}
    lines = [f"{card.notes[i - 1].text} ({why})" for i, why in sorted(v.dropped_notes.items())]
    return {
        "notes": [n.model_dump() for n in survivors],
        "sources": [s.model_dump() for s in card.sources if s.source_id in cited],
        "credibility": {k: r for k, r in credibility.items() if k in cited},
        "verified_notes": len(survivors),
        "dropped": [*card.dropped, *lines],
    }


def _review(board: Board, op: Review, cap: int | None) -> None:
    v = op.verdict
    if v.card_id != op.card_id:
        raise BoardError(f"verdict is for {v.card_id}, op is for {op.card_id}")
    c = _get(board, op.card_id)
    feedback = [*c.critic_feedback, *v.objections]
    known = {s.source_id for s in c.sources}
    # The Critic's own ratings replace the Researcher's for the sources it rated.
    credibility = {**c.credibility, **{k: r for k, r in v.source_ratings.items() if k in known}}
    match v.kind:
        case VerdictKind.APPROVE:
            kept = _prune(c, v, credibility)
            if not kept["notes"]:
                raise BoardError(f"{c.id}: cannot approve a card with no surviving notes")
            board.cards[c.id] = _update(
                c, S.DONE, critic_feedback=feedback, open_objections=[], **kept
            )
        case VerdictKind.REVISE:
            kept = _prune(c, v, credibility)
            board.cards[c.id] = _update(
                c,
                S.IN_PROGRESS,
                critic_feedback=feedback,
                open_objections=v.objections,
                attempts=c.attempts + 1,
                **kept,
            )
        case VerdictKind.REJECT:
            board.cards[c.id] = _update(
                c, S.REJECTED, critic_feedback=feedback, reason="; ".join(v.objections)
            )
        case VerdictKind.SPLIT:
            child_ids = [f"{c.id}.{i}" for i, _ in enumerate(v.split_into, start=1)]
            board.cards[c.id] = _update(
                c,
                S.REJECTED,
                critic_feedback=feedback,
                reason=f"split into {', '.join(child_ids)}",
            )
            for child_id, question in zip(child_ids, v.split_into, strict=True):
                _create(board, child_id, question, c.id, cap)


def apply(board: Board, ops: Sequence[Op], max_cards: int | None = None) -> Board:
    """Return a new board with ops applied. Replayed ops are skipped.

    The whole batch is atomic: if any op fails, the input board is left unchanged
    and nothing is returned. max_cards is the sub-question cap; splits count against it.
    """
    new = board.model_copy(deep=True)
    for op in ops:
        if op.op_id in new.applied_ops:
            continue
        _apply_one(new, op, max_cards)
        new.applied_ops.add(op.op_id)
    return new
