"""
Retrieval service (RAG).

Owns the vector index lifecycle, builds queries from a structured design
brief, and fuses dense and lexical rankings. This is what grounds the agents'
output in verifiable domain facts instead of model memory.

Retrieval is hybrid. Dense vectors alone are a poor fit for the default
hashing embedder, which has no semantic component and scores everything
around 0.2; a brief asking for a "bunk bed" needs the passage that says
exactly that. BM25 supplies exact-term matching, and the two rankings are
combined by Reciprocal Rank Fusion so neither scale has to be calibrated
against the other.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.observability import get_logger, metrics, tracer
from app.core.schemas import DesignRequest, RetrievedContext
from app.services.bm25 import BM25Index, reciprocal_rank_fusion
from app.services.embeddings import get_embedder
from app.services.knowledge_base import get_knowledge_base
from app.services.knowledge_metadata import matches_filter
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
        self._bm25 = BM25Index()
        # The corpus, kept in memory so metadata filtering and citation
        # building need no second lookup.
        self._entries: List[Dict[str, Any]] = []
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
            else:
                # The persisted store skips embedding, but the lexical index
                # and metadata are cheap to rebuild and must stay in step
                # with whatever corpus the store was built from.
                self._build_lexical(get_knowledge_base())

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
                metadata={
                    "index": i,
                    "rooms": entry.get("rooms", []),
                    "styles": entry.get("styles", []),
                    "budget_tiers": entry.get("budget_tiers", []),
                    "vastu": bool(entry.get("vastu", False)),
                },
                embedding=vectors[i],
            )
            for i, entry in enumerate(entries)
        ]
        store.add(chunks)
        store.save()
        self._build_lexical(entries)
        logger.info(
            "kb_index_built chunks=%d dim=%d vocab=%d",
            len(chunks), self._embedder.dim, self._bm25.vocabulary_size,
        )

    def _build_lexical(self, entries: List[Dict[str, Any]]) -> None:
        """
        Build the BM25 index over the same corpus.

        Kept in step with the vector store deliberately: a lexical index
        pointing at different documents would fuse nonsense, and the two are
        rebuilt together in `_build_index`.
        """
        self._bm25 = BM25Index().build([e["content"] for e in entries])
        self._entries = entries

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
        room: Optional[str] = None,
        style: Optional[str] = None,
        budget_tier: Optional[str] = None,
        require_vastu: bool = False,
    ) -> List[RetrievedContext]:
        """
        Retrieve the most relevant knowledge chunks for a query.

        Hybrid: the dense and BM25 rankings are fused with Reciprocal Rank
        Fusion. The two signals are fused by position rather than blended by
        score, because a hashing cosine (~0.2) and a BM25 score (unbounded)
        are on unrelated scales and any weighting between them would be
        arbitrary.

        Filters narrow the candidate pool but never exclude general advice: a
        passage with no room tag is applicable everywhere, so demanding a tag
        would discard the most broadly useful guidance in the corpus.
        """
        if not self._ready:
            self.initialize()
        if self._store is None or self._store.is_empty:
            return []

        top_k = top_k or settings.RAG_TOP_K
        # The fused score is a rank score, not a cosine, so RAG_MIN_SCORE no
        # longer applies here; it stays for the dense-only path.
        pool = max(top_k * 4, 20)

        with tracer.start("rag.retrieve", query=query[:80]):
            qvec = self._embedder.embed(query)
            dense = self._store.search(qvec, top_k=pool, min_score=0.0)
            lexical = self._bm25.top_k(query, pool)

            # Both rankings index the same corpus by position.
            dense_ranked = [
                (int(c.metadata.get("index", 0)), float(s)) for c, s in dense
            ]
            fused = reciprocal_rank_fusion([dense_ranked, lexical])

            entries = self._entries or get_knowledge_base()

            picked: List[RetrievedContext] = []
            for idx, score in fused:
                if idx < 0 or idx >= len(entries):
                    continue
                entry = entries[idx]
                if category and entry.get("category") != category:
                    continue
                if not matches_filter(
                    entry, room=room, style=style,
                    budget_tier=budget_tier, require_vastu=require_vastu,
                ):
                    continue
                picked.append(RetrievedContext(
                    content=entry["content"],
                    source=entry.get("source", ""),
                    score=round(score, 4),
                    metadata={
                        "category": entry.get("category", ""),
                        "rooms": entry.get("rooms", []),
                        "styles": entry.get("styles", []),
                        "budget_tiers": entry.get("budget_tiers", []),
                        "vastu": bool(entry.get("vastu", False)),
                    },
                ))
                if len(picked) >= top_k:
                    break

        metrics.increment("rag.queries")
        metrics.observe("rag.top_score", picked[0].score if picked else 0.0)
        logger.info(
            "rag_retrieve hits=%d top_score=%.3f dense=%d lexical=%d",
            len(picked), picked[0].score if picked else 0.0,
            len(dense), len(lexical),
        )
        return picked

    async def retrieve_for_request(
        self, request: DesignRequest, top_k: Optional[int] = None
    ) -> List[RetrievedContext]:
        """
        Retrieve for a brief, passing its structured fields as filters.

        Room, style, and budget are already known from the form, so spending
        recall on guidance for the wrong kind of room is pure waste.
        """
        return await self.retrieve(
            self.build_query(request),
            top_k=top_k,
            room=request.room_type.value,
            style=request.style.value,
            budget_tier=request.budget.value,
            require_vastu=request.vastu_compliance.value == "strict",
        )

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
