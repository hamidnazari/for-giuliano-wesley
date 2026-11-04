"""web_search tool: Tavily backend with record and replay for deterministic evals."""

import json
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import requests
from pydantic import BaseModel, ConfigDict

from research_assistant.agents.factory import MissingCredential
from research_assistant.guardrails.budget import Budget
from research_assistant.reliability.retry import RetryableError
from research_assistant.settings import SearchConfig

CASSETTE_VERSION = 1


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str = ""
    content: str = ""
    score: float = 0.0


class SearchBackend(Protocol):
    def search(self, query: str, depth: str, max_results: int) -> list[SearchResult]: ...


class ExtractBackend(Protocol):
    def extract(self, url: str) -> str:
        """Return the page's text, or an empty string when it cannot be read."""
        ...


class WebBackend(SearchBackend, ExtractBackend, Protocol):
    pass


class ReplayMiss(Exception):
    """Replay mode saw a query that is not in the cassette."""


class _TavilyLike(Protocol):
    def search(self, query: str, **kwargs: Any) -> dict[str, Any]: ...

    def extract(self, urls: list[str], **kwargs: Any) -> dict[str, Any]: ...


class TavilyBackend:
    def __init__(self, client: _TavilyLike) -> None:
        self._client = client

    def search(self, query: str, depth: str, max_results: int) -> list[SearchResult]:
        try:
            raw = self._client.search(query, search_depth=depth, max_results=max_results)
        except requests.RequestException as e:  # network trouble is worth retrying
            raise RetryableError(repr(e)) from e
        return [
            SearchResult(
                url=r["url"],
                title=r.get("title", ""),
                content=r.get("content", ""),
                score=float(r.get("score", 0.0)),
            )
            for r in raw.get("results", [])
        ]

    def extract(self, url: str) -> str:
        try:
            raw = self._client.extract(urls=[url], extract_depth="basic", format="text")
        except requests.RequestException as e:
            raise RetryableError(repr(e)) from e
        results = raw.get("results", [])
        return str(results[0].get("raw_content") or "") if results else ""


def _key(query: str, depth: str, max_results: int) -> str:
    return f"{depth}|{max_results}|{' '.join(query.lower().split())}"


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": CASSETTE_VERSION, "entries": {}}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != CASSETTE_VERSION:
        raise ValueError(f"unsupported cassette version in {path}")
    return data


class ReplayBackend:
    """Serves recorded results only. Never touches the network."""

    def __init__(self, cassette: Path) -> None:
        data = _read(cassette)
        self._entries: dict[str, Any] = data["entries"]
        self._extracts: dict[str, str] = data.get("extracts", {})

    def search(self, query: str, depth: str, max_results: int) -> list[SearchResult]:
        key = _key(query, depth, max_results)
        if key not in self._entries:
            raise ReplayMiss(f"no recorded search for: {key}")
        return [SearchResult.model_validate(r) for r in self._entries[key]]

    def extract(self, url: str) -> str:
        if url not in self._extracts:
            raise ReplayMiss(f"no recorded page for: {url}")
        return self._extracts[url]


class RecordingBackend:
    """Calls a real backend and saves each response to the cassette."""

    def __init__(self, inner: WebBackend, cassette: Path) -> None:
        self._inner = inner
        self._path = cassette
        self._lock = threading.Lock()

    def search(self, query: str, depth: str, max_results: int) -> list[SearchResult]:
        results = self._inner.search(query, depth, max_results)
        with self._lock:
            data = _read(self._path)
            data["entries"][_key(query, depth, max_results)] = [r.model_dump() for r in results]
            self._write(data)
        return results

    def _write(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)

    def extract(self, url: str) -> str:
        text = self._inner.extract(url)
        with self._lock:
            data = _read(self._path)
            data.setdefault("extracts", {})[url] = text
            self._write(data)
        return text


Mode = Literal["live", "record", "replay"]


def make_backend(
    mode: Mode, cassette: Path | None = None, env: Mapping[str, str] | None = None
) -> WebBackend:
    if mode != "live" and cassette is None:
        raise ValueError(f"{mode} mode needs a cassette path")
    if mode == "replay":
        assert cassette is not None
        return ReplayBackend(cassette)
    env = os.environ if env is None else env
    key = env.get("TAVILY_API_KEY")
    if not key:
        raise MissingCredential("TAVILY_API_KEY is not set")
    from tavily import TavilyClient  # pyright: ignore[reportMissingTypeStubs]

    live = TavilyBackend(cast(_TavilyLike, TavilyClient(api_key=key)))  # untyped library
    if mode == "record":
        assert cassette is not None
        return RecordingBackend(live, cassette)
    return live


def make_web_search(
    backend: SearchBackend, budget: Budget, config: SearchConfig
) -> Callable[..., list[dict[str, Any]]]:
    """The web_search tool function. Charges the search budget before every call."""

    def web_search(query: str) -> list[dict[str, Any]]:
        budget.before_search()
        results = backend.search(query, config.search_depth, config.max_results)
        return [r.model_dump() for r in results]

    return web_search
