"""Append-only JSONL audit log with secret redaction."""

import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research_assistant.models import Event

_SECRET_NAME_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
_MIN_SECRET_LEN = 8


def secrets_from_env(env: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if env is None else env
    return [
        v
        for k, v in env.items()
        if len(v) >= _MIN_SECRET_LEN and any(h in k.upper() for h in _SECRET_NAME_HINTS)
    ]


class AuditLog:
    def __init__(self, path: Path, run_id: str, secrets: list[str] | None = None) -> None:
        self.path = path
        self.run_id = run_id
        self._secrets = secrets if secrets is not None else secrets_from_env()
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, actor: str, type: str, payload: dict[str, Any] | None = None) -> Event:
        event = Event(run_id=self.run_id, actor=actor, type=type, payload=payload or {})
        line = event.model_dump_json()
        for secret in self._secrets:
            line = line.replace(secret, "[REDACTED]")
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return event

    def events(self) -> list[Event]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [Event.model_validate_json(line) for line in f if line.strip()]
