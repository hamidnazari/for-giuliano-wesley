"""Pure mapping from our card statuses to Kanzo's fixed columns and tags."""

from research_assistant.models import CardStatus

# Kanzo columns are fixed: todo, doing, review, done (plus an optional backlog).
# Backlog maps to todo so we do not depend on the optional Backlog setting.
_MAP: dict[CardStatus, tuple[str, tuple[str, ...]]] = {
    CardStatus.BACKLOG: ("todo", ()),
    CardStatus.IN_PROGRESS: ("doing", ()),
    CardStatus.IN_REVIEW: ("review", ()),
    CardStatus.DONE: ("done", ()),
    CardStatus.BLOCKED: ("review", ("blocked",)),
    CardStatus.REJECTED: ("review", ("rejected",)),
}


def to_kanzo(status: CardStatus) -> tuple[str, tuple[str, ...]]:
    """Return (kanzo_status, tags) for a card status."""
    return _MAP[status]
