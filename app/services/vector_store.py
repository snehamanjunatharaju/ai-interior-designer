"""
Vector store: in-memory index with optional disk persistence.

A NumPy-backed index providing cosine-similarity search. Suitable up to
~100k chunks; beyond that, swap the storage layer for FAISS/Qdrant without
touching callers, since the public surface here is `VectorStore`.
"""

from __future__ import annotations

import json
import os
import pickle
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.core.observability import get_logger, metrics, tracer

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A single indexed knowledge-base document fragment."""

    id: str
    content: str
    source: str = ""
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_meta(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("embedding", None)
        return data


class VectorStore:
    """
    Thread-safe cosine-similarity vector index.

    Vectors are L2-normalized on insert, so inner product equals cosine
    similarity and search is a single matmul.
    """

    def __init__(self, dim: int, persist_path: Optional[str] = None) -> None:
        self.dim = dim
        self.persist_path = persist_path
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._chunks: List[Chunk] = []
        self._lock = threading.RLock()

    # -- properties ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    def stats(self) -> Dict[str, Any]:
        if not self._chunks:
            return {"chunks": 0, "dim": self.dim, "categories": {}}
        cats: Dict[str, int] = {}
        for chunk in self._chunks:
            cats[chunk.category] = cats.get(chunk.category, 0) + 1
        return {
            "chunks": len(self._chunks),
            "dim": self.dim,
            "sources": len({c.source for c in self._chunks}),
            "categories": cats,
        }

    # -- mutations ----------------------------------------------------------

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Insert or replace chunks by id."""
        if not chunks:
            return

        incoming = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.id} has no embedding")
            vec = np.asarray(chunk.embedding, dtype=np.float32)
            if vec.shape[0] != self.dim:
                raise ValueError(
                    f"embedding dim mismatch: got {vec.shape[0]}, expected {self.dim}"
                )
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            chunk.embedding = vec
            incoming.append(chunk)

        with self._lock:
            incoming_ids = {c.id for c in incoming}
            # Drop any existing version of the same ids to keep upserts clean.
            if self._chunks and incoming_ids:
                keep = [i for i, c in enumerate(self._chunks) if c.id not in incoming_ids]
                if len(keep) != len(self._chunks):
                    self._vectors = self._vectors[keep] if keep else np.zeros(
                        (0, self.dim), dtype=np.float32
                    )
                    self._chunks = [self._chunks[i] for i in keep]

            self._vectors = (
                np.vstack([self._vectors, np.vstack([c.embedding for c in incoming])])
                if len(self._vectors)
                else np.vstack([c.embedding for c in incoming])
            )
            self._chunks.extend(incoming)

        metrics.gauge("vectorstore.chunks", len(self._chunks))
        logger.info("vectorstore_add count=%d total=%d", len(incoming), len(self._chunks))

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._vectors = np.zeros((0, self.dim), dtype=np.float32)

    # -- queries ------------------------------------------------------------

    def search(
        self, query_vector: np.ndarray, top_k: int = 4, min_score: float = 0.0
    ) -> List[tuple[Chunk, float]]:
        """Return the top-k most similar chunks above `min_score`."""
        with self._lock:
            if not self._chunks:
                return []
            query = np.asarray(query_vector, dtype=np.float32)
            if query.shape[0] != self.dim:
                raise ValueError("query vector dim mismatch")
            norm = float(np.linalg.norm(query))
            if norm > 0:
                query = query / norm

            scores = self._vectors @ query
            k = min(top_k, len(scores))
            # argpartition avoids a full sort: O(n) instead of O(n log n).
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

            return [
                (self._chunks[i], float(scores[i]))
                for i in top_idx
                if scores[i] >= min_score
            ]

    def get(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            for chunk in self._chunks:
                if chunk.id == chunk_id:
                    return chunk
        return None

    # -- persistence --------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.persist_path
        if not path:
            return
        with self._lock:
            payload = {
                "dim": self.dim,
                "chunks": [
                    {**c.to_meta(), "embedding": c.embedding.tolist()}
                    for c in self._chunks
                ],
            }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)  # Atomic swap: never leaves a corrupt index.
        logger.info("vectorstore_saved path=%s chunks=%d", path, len(payload["chunks"]))

    @classmethod
    def load(cls, dim: int, path: str) -> "VectorStore":
        store = cls(dim=dim, persist_path=path)
        if not os.path.exists(path):
            logger.info("vectorstore_load miss path=%s", path)
            return store
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            chunks = [
                Chunk(
                    id=item["id"],
                    content=item["content"],
                    source=item.get("source", ""),
                    category=item.get("category", ""),
                    metadata=item.get("metadata", {}),
                    embedding=np.asarray(item["embedding"], dtype=np.float32),
                )
                for item in payload.get("chunks", [])
            ]
            store.add(chunks)
            logger.info("vectorstore_loaded path=%s chunks=%d", path, len(chunks))
        except Exception as exc:
            logger.error("vectorstore_load_failed path=%s error=%s", path, exc)
        return store
