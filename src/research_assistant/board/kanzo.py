"""Mirror the board into Kanzo over MCP (stdio). Kanzo is a view; the graph state is the truth."""

import asyncio
import re
import threading
from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path
from types import TracebackType
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from research_assistant.board.ops import Block, Op, Reject, Review, SubmitWork
from research_assistant.board.render import card_markdown
from research_assistant.board.sink import BoardSink
from research_assistant.kanzo_mapping import to_kanzo
from research_assistant.models import Board, Card
from research_assistant.observability.audit import AuditLog

_CREATED = re.compile(r"^Created (\S+#\d+) ")


class KanzoError(Exception):
    pass


class _Bridge:
    """Runs one MCP session on its own thread so synchronous code can call tools.

    The session lives in a single long-running task: anyio scopes must be exited by the
    task that entered them, so every call is passed to that task through a queue.
    """

    def __init__(self, params: StdioServerParameters) -> None:
        self._params = params
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._ready: Future[None] = Future()
        self._queue: asyncio.Queue[tuple[str, dict[str, Any], Future[str]] | None] | None = None
        self._main: Future[None] | None = None

    def start(self, timeout: float = 30.0) -> None:
        self._thread.start()
        self._main = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        self._ready.result(timeout=timeout)

    async def _serve(self) -> None:
        self._queue = asyncio.Queue()
        try:
            async with stdio_client(self._params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._ready.set_result(None)
                    while (item := await self._queue.get()) is not None:
                        name, args, fut = item
                        try:
                            res = await session.call_tool(name, args)
                            text = "".join(getattr(c, "text", "") for c in res.content)
                            if res.is_error:
                                fut.set_exception(KanzoError(f"{name}: {text}"))
                            else:
                                fut.set_result(text)
                        except Exception as e:
                            fut.set_exception(e)
        except Exception as e:
            if not self._ready.done():
                self._ready.set_exception(e)
            raise

    def call(self, name: str, args: dict[str, Any], timeout: float = 30.0) -> str:
        if self._main is None or self._main.done():
            raise KanzoError("kanzo session is not running")
        fut: Future[str] = Future()

        async def put() -> None:
            assert self._queue is not None
            await self._queue.put((name, args, fut))

        asyncio.run_coroutine_threadsafe(put(), self._loop).result(timeout=timeout)
        return fut.result(timeout=timeout)

    def stop(self) -> None:
        async def put_stop() -> None:
            assert self._queue is not None
            await self._queue.put(None)

        if self._main is not None and self._queue is not None:
            asyncio.run_coroutine_threadsafe(put_stop(), self._loop).result(timeout=10)
            self._main.result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)


class KanzoSink:
    """Pushes card changes to a Kanzo project, one project per run."""

    def __init__(self, project_slug: str, home: Path | None = None, command: str = "kanzo") -> None:
        args = (["--home", str(home)] if home else []) + ["mcp"]
        self._slug = project_slug
        self._bridge = _Bridge(StdioServerParameters(command=command, args=args))
        self._ids: dict[str, str] = {}  # our card id -> kanzo task id (like "slug#3")

    def __enter__(self) -> "KanzoSink":
        self._bridge.start()
        self._bridge.call("create_project", {"slug": self._slug, "name": f"Research {self._slug}"})
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._bridge.stop()

    def call(self, tool: str, **args: Any) -> str:
        """Raw MCP call, for tests and debugging."""
        return self._bridge.call(tool, args)

    def task_id(self, card_id: str) -> str:
        return self._ids[card_id]

    def sync(self, board: Board, ops: Sequence[Op]) -> None:
        for op in ops:
            for card in self._affected(board, op):
                self._push(card)
            self._comment(op)

    def _affected(self, board: Board, op: Op) -> list[Card]:
        """The op's card, plus any children a split just created."""
        ids = [op.card_id]
        ids += [c.id for c in board.cards.values() if c.parent_id == op.card_id]
        return [board.cards[i] for i in ids if i in board.cards]

    def _push(self, card: Card) -> None:
        status, tags = to_kanzo(card.status)
        title = f"{card.id}: {card.sub_question}"
        common = {"status": status, "tags": ",".join(tags), "description": card_markdown(card)}
        if card.id not in self._ids:
            text = self._bridge.call(
                "create_task", {"project": self._slug, "title": title, **common}
            )
            match = _CREATED.match(text)
            if not match:
                raise KanzoError(f"unexpected create_task reply: {text!r}")
            self._ids[card.id] = match.group(1)
        else:
            self._bridge.call("update_task", {"id": self._ids[card.id], "title": title, **common})

    def _comment(self, op: Op) -> None:
        task = self._ids.get(op.card_id)
        if task is None:
            return
        if isinstance(op, Review):
            v = op.verdict
            body = f"Verdict: {v.kind.value} (rubric {v.rubric_version})"
            if v.objections:
                body += "\n" + "\n".join(f"- {o}" for o in v.objections)
            self._bridge.call("add_comment", {"id": task, "body": body, "author": "critic"})
        elif isinstance(op, Block | Reject):
            self._bridge.call(
                "add_comment", {"id": task, "body": op.reason, "author": "orchestrator"}
            )
        elif isinstance(op, SubmitWork):
            body = f"Submitted {len(op.notes)} notes with {len(op.sources)} sources."
            self._bridge.call("add_comment", {"id": task, "body": body, "author": "researcher"})


class SafeSink:
    """Wraps a sink so a mirror failure is logged, not fatal. The board is only a view."""

    def __init__(self, inner: BoardSink, audit: AuditLog) -> None:
        self._inner = inner
        self._audit = audit

    def sync(self, board: Board, ops: Sequence[Op]) -> None:
        try:
            self._inner.sync(board, ops)
        except Exception as e:
            self._audit.record("board", "sink_error", {"error": repr(e)})
