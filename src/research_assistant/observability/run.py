"""Run ID and per-run project folder."""

import secrets
from datetime import UTC, datetime
from pathlib import Path


def new_run_id(now: datetime | None = None) -> str:
    """Sortable, unique run ID such as 20261105-071530-a1b2c3."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def project_dir(data_root: Path, run_id: str) -> Path:
    """Create and return data/projects/<run-id>/."""
    path = data_root / "projects" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path
