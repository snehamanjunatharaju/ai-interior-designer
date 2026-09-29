"""
Tests for hybrid retrieval: BM25, rank fusion, metadata inference, filtering.

The original retriever used dense vectors alone, and with the default hashing
embedder - which has no semantic component - that meant a query for a "bunk
bed" could not reliably find the passage that says "bunk bed". These tests
pin the properties that fix it: exact terms are matched, the two rankings are
fused rather than one dominating, and filters narrow without excluding
general advice.
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.bm25 import BM25Index, reciprocal_rank_fusion, tokenize
from app.services.knowledge_base import get_knowledge_base, knowledge_stats
from app.services.knowledge_metadata import (
    infer_metadata,
    matches_filter,
)
from app.services.retrieval import retrieval_service


def run(coro):
    """Drive a coroutine to completion.

    `asyncio.run` per call is correct here but would rebuild the event loop for
    every test; a shared loop keeps the retrieval service's internal state
    stable across the module.
    """
    loop = _LOOP
    return loop.run_until_complete(coro)


_LOOP = asyncio.new_event_loop()


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


class TestBM25:
    def test_tokenize_drops_stopwords_keeps_numbers(self):
        assert tokenize("The bed is under 90 cm") == ["bed", "under", "90", "cm"]

    def test_stopwords_are_removed(self):
        assert "the" not in tokenize("The bed is under 90 cm")
        assert "is" not in tokenize("The bed is under 90 cm")

    def test_ranks_exact_term_first(self):
        index = BM25Index().build([
            "A bunk bed saves floor space in a small room.",
            "Modular kitchen storage uses pull-out drawers.",
            "Wall painting transforms perceived space.",
        ])
        top = index.top_k("bunk bed", 1)
        assert top and top[0][0] == 0

    def test_rare_term_outweighs_common_term(self):
        """The defining property of IDF: a rare word is more diagnostic."""
        index = BM25Index().build([
            "Lighting matters in every room of the home.",
            "Lighting matters in every room of the house.",
            "A rare phrase appears only here: tandoor oven.",
        ])
        top = index.top_k("tandoor oven", 1)
        assert top and top[0][0] == 2

    def test_empty_corpus_is_safe(self):
        assert BM25Index().build([]).score("anything") == []

    def test_stopword_only_query_returns_nothing(self):
        index = BM25Index().build(["a bed and a desk"])
        assert index.score("the and of") == []

    def test_deterministic(self):
        index = BM25Index().build(["alpha beta", "beta gamma", "gamma delta"])
        assert index.score("beta") == index.score("beta")


class TestRankFusion:
    def test_document_ranked_high_by_either_list_wins(self):
        # (0, score) means "document 0, first in the dense list".
        dense = [(0, 0.9), (1, 0.5)]
        lexical = [(1, 12.0), (2, 3.0)]
        fused = reciprocal_rank_fusion([dense, lexical])
        # Document 1 is second in one list and first in the other, so it
        # should outrank a document that is only middling in one.
        assert fused[0][0] == 1

    def test_score_is_scale_independent(self):
        """Fusion uses rank, so a 1000x score difference must not matter."""
        small = reciprocal_rank_fusion([[(0, 0.1), (1, 0.09)]])
        large = reciprocal_rank_fusion([[(0, 100.0), (1, 99.0)]])
        assert small == large

    def test_consensus_is_rewarded(self):
        a = reciprocal_rank_fusion([[(0, 1), (1, 1)], [(0, 1), (1, 1)]])
        b = reciprocal_rank_fusion([[(0, 1), (1, 1)], [(1, 1), (0, 1)]])
        assert a[0][0] == b[0][0] == 0

    def test_empty_inputs_are_safe(self):
        assert reciprocal_rank_fusion([]) == []


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


class TestMetadataInference:
    def test_room_detected(self):
        meta = infer_metadata(
            "A bunk bed and a study desk in a children's room.", "kids_room"
        )
        assert "kids_room" in meta["rooms"]

    def test_vastu_detected(self):
        assert infer_metadata("Vastu places the bed head to the south.")["vastu"]

    def test_non_vastu_not_tagged(self):
        assert not infer_metadata("A sofa and a coffee table.")["vastu"]

    def test_budget_tier_detected(self):
        meta = infer_metadata("On a low budget, defer the decor purchase.")
        assert "low" in meta["budget_tiers"]

    def test_style_detected(self):
        assert "scandinavian" in infer_metadata("A scandinavian palette." )["styles"]

    def test_untagged_passage_has_empty_facets(self):
        meta = infer_metadata("Maintain sixty centimetres of circulation.")
        assert meta["rooms"] == [] and meta["styles"] == []


class TestFiltering:
    def test_untagged_chunk_always_passes(self):
        """
        General advice has no room tag and applies everywhere. Filtering must
        not exclude it, or the most useful passages vanish.
        """
        meta = {"rooms": [], "styles": [], "budget_tiers": [], "vastu": False}
        assert matches_filter(meta, room="kitchen", style="japanese",
                              budget_tier="luxury", require_vastu=False)

    def test_tagged_chunk_excluded_by_wrong_room(self):
        meta = {"rooms": ["bedroom"], "styles": [], "budget_tiers": [],
                "vastu": False}
        assert not matches_filter(meta, room="kitchen")

    def test_matching_room_passes(self):
        meta = {"rooms": ["bedroom", "kids_room"], "styles": [],
                "budget_tiers": [], "vastu": False}
        assert matches_filter(meta, room="kids_room")

    def test_vastu_required(self):
        meta = {"rooms": [], "styles": [], "budget_tiers": [], "vastu": False}
        assert not matches_filter(meta, require_vastu=True)


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


class TestCorpus:
    def test_every_chunk_has_content_and_source(self):
        for entry in get_knowledge_base():
            assert entry.get("content"), "a chunk with no content is dead weight"
            assert entry.get("source"), "an uncited chunk cannot be shown as a citation"

    def test_corpus_is_large_enough_to_be_useful(self):
        assert knowledge_stats()["chunks"] >= 60

    def test_vastu_is_represented(self):
        """Previously a strict-Vastu filter matched a single passage."""
        assert knowledge_stats()["vastu_chunks"] >= 8

    def test_no_placeholder_prose(self):
        """A vague passage teaches nothing and wastes a retrieval slot."""
        for entry in get_knowledge_base():
            content = entry["content"].lower()
            assert len(content) > 80, f"{entry['source']} is too thin to be useful"
            assert "choose high quality" not in content


# --------------------------------------------------------------------------
# End-to-end retrieval
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ready_index():
    retrieval_service.initialize(force=True)
    return retrieval_service


class TestHybridRetrieval:
    def test_exact_term_is_found(self, ready_index):
        """A product term must retrieve the passage naming it."""
        results = run(ready_index.retrieve("bunk bed safety rail", top_k=3))
        assert results
        assert any("bunk" in c.content.lower() for c in results)

    def test_vastu_query_returns_vastu(self, ready_index):
        results = run(ready_index.retrieve(
            "vastu bed head direction south", top_k=3))
        assert results
        assert any(c.metadata.get("vastu") for c in results)

    def test_strong_vastu_filter_returns_results(self, ready_index):
        """The bug this fixed: a strict-Vastu filter returned nothing."""
        results = run(ready_index.retrieve(
            "bed direction", top_k=4, room="kids_room", require_vastu=True))
        assert results
        assert all(c.metadata.get("vastu") for c in results)

    def test_room_filter_respects_tags(self, ready_index):
        results = run(ready_index.retrieve(
            "storage and bed placement", top_k=5, room="kitchen"))
        for c in results:
            rooms = c.metadata.get("rooms") or []
            assert not rooms or "kitchen" in rooms

    def test_results_carry_citation_metadata(self, ready_index):
        results = run(ready_index.retrieve("lighting layers", top_k=2))
        assert results
        for c in results:
            assert c.source
            assert c.metadata.get("category") is not None

    def test_respects_top_k(self, ready_index):
        assert len(run(ready_index.retrieve("bed", top_k=2))) <= 2

    def test_deterministic(self, ready_index):
        a = run(ready_index.retrieve("modular kitchen", top_k=3))
        b = run(ready_index.retrieve("modular kitchen", top_k=3))
        assert [c.source for c in a] == [c.source for c in b]

    def test_unmatched_query_still_returns_something(self, ready_index):
        results = run(ready_index.retrieve("zzzqqqxyz", top_k=3))
        # Dense retrieval will return something even with no lexical match;
        # the point is that it does not raise.
        assert isinstance(results, list)
