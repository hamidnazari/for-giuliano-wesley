"""docs_retrieve: search the user's indexed documents. Read-only and local."""

from collections.abc import Callable
from typing import Any

from research_assistant.documents.index import DocIndex


def make_docs_retrieve(
    index: DocIndex | None, top_k: int, min_score: float
) -> Callable[..., list[dict[str, Any]]]:
    def docs_retrieve(query: str) -> list[dict[str, Any]]:
        if index is None:
            return []
        return [
            {
                "chunk_id": h.chunk.chunk_id,
                "doc": h.chunk.doc,
                "page": h.chunk.page,
                "text": h.chunk.text,
                "score": h.score,
            }
            for h in index.search(query, top_k, min_score)
        ]

    return docs_retrieve
