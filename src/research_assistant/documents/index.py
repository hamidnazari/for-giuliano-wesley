"""A small local vector index: chunks, embeddings, cosine search. Chunk-level citations."""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from research_assistant.documents.chunker import chunk_text
from research_assistant.documents.loader import load_documents

INDEX_VERSION = 1
Vectors = NDArray[np.float32]


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> Vectors:
        """Return an (n, dim) array of L2-normalised vectors."""
        ...


class FastEmbedder:
    """Local ONNX embeddings. The model downloads once, on first use."""

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model: Any = None

    def embed(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.name)
        out = np.vstack(list(self._model.embed(list(texts)))).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class HashingEmbedder:
    """Deterministic word-hash embeddings with no model. For tests: shared words score higher."""

    name = "hashing-test"

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
                out[i, h % self._dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms == 0, 1.0, norms)).astype(np.float32)


@dataclass(frozen=True)
class DocChunk:
    chunk_id: str  # like "notes/a.md#c3"; the citation target
    doc: str
    page: int | None
    text: str

    @property
    def title(self) -> str:
        return f"{self.doc} (page {self.page})" if self.page else self.doc


@dataclass(frozen=True)
class DocHit:
    chunk: DocChunk
    score: float


class IndexUnusable(Exception):
    """The saved index cannot be used with this embedder or is damaged."""


class DocIndex:
    def __init__(
        self, chunks: list[DocChunk], vectors: Vectors, embedder: Embedder, meta: dict[str, Any]
    ) -> None:
        self.chunks = chunks
        self._vectors = vectors
        self._embedder = embedder
        self.meta = meta

    @classmethod
    def build(
        cls, root: Path, embedder: Embedder, chunk_size: int, overlap: int
    ) -> tuple["DocIndex", list[str]]:
        """Index a file or folder. Returns the index and the list of skipped files."""
        loaded = load_documents(root)
        chunks: list[DocChunk] = []
        counts: dict[str, int] = {}
        for page in loaded.pages:
            for piece in chunk_text(page.text, chunk_size, overlap):
                counts[page.doc] = counts.get(page.doc, 0) + 1
                chunks.append(
                    DocChunk(f"{page.doc}#c{counts[page.doc]}", page.doc, page.page, piece)
                )
        vectors = embedder.embed([c.text for c in chunks])
        meta = {
            "version": INDEX_VERSION,
            "model": embedder.name,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "documents": len(counts),
            "created": datetime.now(UTC).isoformat(),
        }
        return cls(chunks, vectors, embedder, meta), loaded.skipped

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(asdict(c)) + "\n" for c in self.chunks)
        np.save(directory / "vectors.npy", self._vectors, allow_pickle=False)
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> "DocIndex":
        try:
            meta: dict[str, Any] = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            vectors: Vectors = np.load(directory / "vectors.npy", allow_pickle=False)
            with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
                chunks = [DocChunk(**json.loads(line)) for line in f if line.strip()]
        except (OSError, ValueError, TypeError, KeyError) as e:
            raise IndexUnusable(f"cannot read index in {directory}: {e}") from e
        if meta.get("version") != INDEX_VERSION:
            raise IndexUnusable(f"index version {meta.get('version')} is not supported")
        if meta.get("model") != embedder.name:
            raise IndexUnusable(
                f"index was built with '{meta.get('model')}' but config uses '{embedder.name}'; "
                "run the index command again"
            )
        if len(chunks) != len(vectors):
            raise IndexUnusable("index files do not match; run the index command again")
        return cls(chunks, vectors, embedder, meta)

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[DocHit]:
        if not self.chunks:
            return []
        q = self._embedder.embed([query])[0]
        scores = self._vectors @ q
        order = [int(i) for i in np.argsort(-scores)[:k]]
        return [DocHit(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]
