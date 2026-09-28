"""
Retrieval service (RAG).

Owns the vector index lifecycle, builds queries from a structured design
brief, and reranks results. This is what grounds the agents' output in
verifiable domain facts instead of model memory.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.observability import get_logger, metrics, tracer
from app.core.schemas import DesignRequest, RetrievedContext
from app.services.embeddings import get_embedder
from app.services.knowledge_base import get_knowledge_base
from app.services.vector_store import Chunk, VectorStore

logger = get_logger(__name__)


class RetrievalService:
    """
    Builds and queries the design knowledge index.

    Index construction is done once at startup and cached to disk, so
    subsequent boots skip embedding entirely.
    """

    def __init__(self) -> None:
        self._embedder = None
        self._store: Optional[VectorStore] = None
        self._lock = threading.Lock()
        self._ready = False

    # -- lifecycle ---------------------------------------------------------

    def initialize(self, force: bool = False) -> None:
        """Load or build the vector index. Idempotent."""
        with self._lock:
            if self._ready and not force:
                return

            self._embedder = get_embedder()
            dim = self._embedder.dim

            store = VectorStore.load(dim, settings.VECTOR_STORE_PATH)
            if store.is_empty or force:
                self._build_index(store)

            self._store = store
            self._ready = True
            logger.info("retrieval_ready stats=%s", store.stats())

    def _build_index(self, store: VectorStore) -> None:
        """Embed the knowledge base and persist the index."""
        entries = get_knowledge_base()
        texts = [e["content"] for e in entries]
        vectors = self._embedder.embed_batch(texts)

        chunks = [
            Chunk(
                id=f"kb-{i:03d}",
                content=entry["content"],
                source=entry.get("source", ""),
                category=entry.get("category", ""),
                metadata={"index": i},
                embedding=vectors[i],
            )
            for i, entry in enumerate(entries)
        ]
        store.add(chunks)
        store.save()
        logger.info("kb_index_built chunks=%d dim=%d", len(chunks), self._embedder.dim)

    # -- query construction ------------------------------------------------

    def build_query(self, request: DesignRequest) -> str:
        """
        Turn a structured brief into a retrieval query.

        A crafted query (rather than raw free text) measurably improves
        recall because the index vocabulary is domain-specific.
        """
        parts = [
            request.room_type.value.replace("_", " "),
            request.style.value.replace("_", " "),
            request.client_preferences,
            request.color_scheme,
        ]
        if request.room_dimensions:
            parts.append(request.room_dimensions)
        if request.furniture:
            parts.append("furniture " + " ".join(request.furniture))
        if request.specific_requirements:
            parts.append(" ".join(request.specific_requirements))
        if request.occupants:
            parts.append("used by " + " ".join(request.occupants))
        if request.lifestyle_notes:
            parts.append(request.lifestyle_notes)
        return " ".join(p for p in parts if p)

    # -- retrieval ---------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        category: Optional[str] = None,
    ) -> List[RetrievedContext]:
        """Retrieve the most relevant knowledge chunks for a query."""
        if not self._ready:
            self.initialize()
        if self._store is None or self._store.is_empty:
            return []

        top_k = top_k or settings.RAG_TOP_K
        min_score = settings.RAG_MIN_SCORE if min_score is None else min_score

        with tracer.start("rag.retrieve", query=query[:80]):
            qvec = self._embedder.embed(query)
            hits = self._store.search(qvec, top_k=top_k * 2, min_score=min_score)

            if category:
                filtered = [(c, s) for c, s in hits if c.category == category]
                hits = filtered or hits

            hits = hits[:top_k]
            if settings.RERANK_ENABLED:
                hits = self._rerank(query, hits)

            results = [
                RetrievedContext(
                    content=chunk.content,
                    source=chunk.source,
                    score=round(score, 4),
                    metadata={**chunk.metadata, "category": chunk.category},
                )
                for chunk, score in hits
            ]

        metrics.increment("rag.queries")
        metrics.observe("rag.top_score", results[0].score if results else 0.0)
        logger.info("rag_retrieve hits=%d top_score=%.3f", len(results),
                    results[0].score if results else 0.0)
        return results

    async def retrieve_for_request(
        self, request: DesignRequest, top_k: Optional[int] = None
    ) -> List[RetrievedContext]:
        """Convenience wrapper that builds the query from a design brief."""
        return await self.retrieve(self.build_query(request), top_k=top_k)

    # -- reranking ---------------------------------------------------------

    def _rerank(
        self, query: str, hits: List[tuple[Chunk, float]]
    ) -> List[tuple[Chunk, float]]:
        """
        Lexical reranking on top of dense retrieval.

        Dense retrieval captures semantics but can miss exact keyword
        matches like a brand name or a number. This blends in token
        overlap, which is cheap and measurably improves precision.
        """
        query_tokens = {t.lower() for t in query.split() if len(t) > 3}
        if not query_tokens:
            return hits

        reranked = []
        for chunk, score in hits:
            chunk_tokens = {t.lower() for t in chunk.content.split() if len(t) > 3}
            overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            # 80% semantic + 20% lexical.
            reranked.append((chunk, score * 0.8 + overlap * 0.2))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    # -- introspection -----------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        if not self._ready:
            self.initialize()
        return {
            **(self._store.stats() if self._store else {}),
            "embedder": self._embedder.name if self._embedder else "uninitialized",
            "top_k": settings.RAG_TOP_K,
            "rerank_enabled": settings.RERANK_ENABLED,
        }

    def format_context(self, contexts: List[RetrievedContext], max_chars: int = 3000) -> str:
        """Render retrieved context as a prompt-ready block with citations."""
        if not contexts:
            return ""
        lines = []
        for i, ctx in enumerate(contexts, 1):
            lines.append(f"[{i}] (relevance {ctx.score:.2f}) {ctx.content}")
        text = "\n".join(lines)
        return text[:max_chars]


retrieval_service = RetrievalService()
