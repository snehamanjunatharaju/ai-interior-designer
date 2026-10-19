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
            # A persisted store built from an older corpus has to be rebuilt.
            # Otherwise passages added later are in the BM25 index but not the
            # vectors, so half the retriever cannot see them.
            corpus_size = len(get_knowledge_base())
            stored = int((store.stats() or {}).get("chunks") or 0)
            if store.is_empty or force or stored != corpus_size:
                logger.info(
                    "kb_rebuild stored=%d corpus=%d", stored, corpus_size,
                )
                store = VectorStore(dim, settings.VECTOR_STORE_PATH)
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
        pool: Optional[int] = None,
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
        # An explicit `top_k` is honoured (an API caller asking for 2 gets 2).
        # Only the request-wide retrieve applies the 3-5 band.
        # The fused score is a rank score, not a cosine, so RAG_MIN_SCORE no
        # longer applies here; it stays for the dense-only path.
        pool = pool or max(top_k * 4, 20)

        with tracer.start("rag.retrieve", query=query[:80]):
            qvec = self._embedder.embed(query)
            dense = self._store.search(qvec, top_k=pool, min_score=0.0)
            lexical = self._bm25.top_k(query, pool)

            # Both rankings index the same corpus by position.
            dense_ranked = [
                (int(c.metadata.get("index", 0)), float(s)) for c, s in dense
            ]
            dense_similarity = {idx: score for idx, score in dense_ranked}
            lexical_score = {idx: score for idx, score in lexical}
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
                        # `score` is the fused RRF rank score (top hit ~1/61),
                        # which is not a similarity and must not be presented
                        # as one. The real cosine and BM25 figures are here.
                        "dense_similarity": round(dense_similarity.get(idx, 0.0), 4),
                        "lexical_score": round(lexical_score.get(idx, 0.0), 3),
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

        Returns 3-5 passages. One passage at 0.014 similarity is not grounding:
        the architect then wrote from model memory and the citation was
        decoration. When the filters are stricter than the corpus, the filters
        are relaxed until the floor is reached - a passage with no room tag is
        general advice and still applies.
        """
        target = max(
            settings.RAG_MIN_PASSAGES,
            min(top_k or settings.RAG_TOP_K, settings.RAG_MAX_PASSAGES),
        )
        # The relevant corpus is small (a handful of kitchen passages), so the
        # candidate pool has to be wide enough that relaxing a filter can
        # actually find them rather than settling on whatever was already in
        # the top four.
        pool = max(target * 8, 40)
        filters = dict(
            room=request.room_type.value,
            style=request.style.value,
            budget_tier=request.budget.value,
            require_vastu=request.vastu_compliance.value == "strict",
        )
        query = self.build_query(request)
        results = await self.retrieve(
            query, top_k=target, **filters, pool=pool
        )
        if len(results) >= min(target, settings.RAG_MIN_PASSAGES):
            return results

        # A genuinely relevant passage for this room outranks a generic one,
        # so a room match is tried without the style and budget tags first.
        by_room = await self.retrieve(
            query, top_k=target, room=filters["room"], pool=pool
        )
        if len(by_room) > len(results):
            results = by_room
        if len(results) >= min(target, settings.RAG_MIN_PASSAGES):
            return results

        picked_content = {c.content for c in results}
        # Relax the most restrictive filters first, one at a time. The earlier
        # version relaxed style before budget, and a kitchen brief whose style
        # matched nothing left the room filter in place until the last
        # fallback - which is why it came back with two passages, neither
        # about kitchens. Budget is the least important filter for grounding.
        for relaxed in (
            {**filters, "require_vastu": False},
            {**filters, "require_vastu": False, "budget_tier": None},
            {**filters, "budget_tier": None, "style": None},
            {**filters, "require_vastu": False, "budget_tier": None, "style": None},
            {},
        ):
            extra = await self.retrieve(query, top_k=target, **relaxed, pool=pool)
            for ctx in extra:
                if ctx.content in picked_content:
                    continue
                ctx.metadata = {**ctx.metadata, "filter_relaxed": True}
                results.append(ctx)
                picked_content.add(ctx.content)
                if len(results) >= target:
                    return results
        return results

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
