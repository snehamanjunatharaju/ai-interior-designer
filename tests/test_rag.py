"""
Tests for the RAG stack: embeddings, vector store, and retrieval.

No LLM calls — the embedder is deterministic and local, so these run fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.embeddings import HashingEmbedder, get_embedder
from app.services.knowledge_base import get_knowledge_base
from app.services.llm_provider import extract_json
from app.services.vector_store import Chunk, VectorStore


class TestEmbeddings:
    def test_vector_shape_and_norm(self):
        emb = HashingEmbedder(dim=64)
        vec = emb.embed("small compact living room")
        assert vec.shape == (64,)
        # Vectors are L2-normalized, so cosine sim == dot product.
        assert pytest.approx(float(np.linalg.norm(vec)), abs=1e-5) == 1.0

    def test_deterministic_across_instances(self):
        a = HashingEmbedder(dim=128).embed("vastu compliant kitchen")
        b = HashingEmbedder(dim=128).embed("vastu compliant kitchen")
        assert np.allclose(a, b)

    def test_similar_text_scores_higher_than_unrelated(self):
        emb = HashingEmbedder(dim=256)
        query = emb.embed("small budget living room apartment")
        near = emb.embed("small living room for a compact apartment")
        far = emb.embed("industrial pipeline valve calibration")
        assert float(query @ near) > float(query @ far)

    def test_synonym_expansion_improves_match(self):
        """'couch' and 'sofa' should land close together."""
        emb = HashingEmbedder(dim=384)
        sofa = emb.embed("a sofa for the living room")
        couch = emb.embed("a couch for the living room")
        tv = emb.embed("a television for the living room")
        assert float(sofa @ couch) > float(sofa @ tv)

    def test_empty_text_is_zero_vector(self):
        vec = HashingEmbedder(dim=32).embed("!!! ???")
        assert np.allclose(vec, 0.0)

    def test_batch_matches_individual(self):
        emb = HashingEmbedder(dim=64)
        texts = ["one", "two", "three"]
        batch = emb.embed_batch(texts)
        for i, t in enumerate(texts):
            assert np.allclose(batch[i], emb.embed(t))

    def test_get_embedder_singleton(self):
        assert get_embedder() is get_embedder()


class TestVectorStore:
    @pytest.fixture
    def store(self) -> VectorStore:
        s = VectorStore(dim=32)
        emb = HashingEmbedder(dim=32)
        texts = [
            ("c1", "small living room sofa layout"),
            ("c2", "modular kitchen granite countertop"),
            ("c3", "master bedroom wardrobe storage"),
            ("c4", "vastu kitchen stove south east"),
        ]
        s.add([
            Chunk(id=i, content=t, source="test", category="test", embedding=emb.embed(t))
            for i, t in texts
        ])
        return s

    def test_add_and_len(self, store: VectorStore):
        assert len(store) == 4
        assert not store.is_empty

    def test_search_ranks_relevant_first(self, store: VectorStore):
        emb = HashingEmbedder(dim=32)
        hits = store.search(emb.embed("modular kitchen countertop granite"), top_k=2)
        assert hits
        assert hits[0][0].id == "c2"
        assert hits[0][1] > hits[1][1]

    def test_min_score_filters(self, store: VectorStore):
        emb = HashingEmbedder(dim=32)
        hits = store.search(emb.embed("quantum entanglement"), top_k=4, min_score=0.9)
        assert hits == []

    def test_upsert_replaces_by_id(self):
        s = VectorStore(dim=32)
        emb = HashingEmbedder(dim=32)
        s.add([Chunk(id="a", content="original text", embedding=emb.embed("original text"))])
        s.add([Chunk(id="a", content="replaced text", embedding=emb.embed("replaced text"))])
        assert len(s) == 1
        assert s.get("a").content == "replaced text"

    def test_dimension_mismatch_rejected(self):
        s = VectorStore(dim=16)
        with pytest.raises(ValueError):
            s.add([Chunk(id="a", content="x", embedding=np.ones(32, dtype=np.float32))])

    def test_empty_store_returns_nothing(self):
        s = VectorStore(dim=16)
        assert s.search(np.ones(16, dtype=np.float32)) == []

    def test_save_load_roundtrip(self, tmp_path, store: VectorStore):
        path = str(tmp_path / "vs.json")
        store.save(path)
        loaded = VectorStore.load(32, path)
        assert len(loaded) == 4
        assert loaded.get("c1").content == store.get("c1").content

    def test_load_missing_path_is_empty(self, tmp_path):
        loaded = VectorStore.load(16, str(tmp_path / "nope.json"))
        assert loaded.is_empty

    def test_stats(self, store: VectorStore):
        stats = store.stats()
        assert stats["chunks"] == 4
        assert stats["dim"] == 32


class TestRetrieval:
    @pytest.mark.asyncio
    async def test_returns_ranked_contexts(self, sample_request):
        from app.services.retrieval import retrieval_service

        retrieval_service.initialize()
        contexts = await retrieval_service.retrieve_for_request(sample_request)
        assert contexts
        # Scores must be monotonically non-increasing.
        scores = [c.score for c in contexts]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_budget_query_surfaces_budget_knowledge(self, sample_request):
        from app.services.retrieval import retrieval_service

        retrieval_service.initialize()
        contexts = await retrieval_service.retrieve_for_request(sample_request)
        blob = " ".join(c.content.lower() for c in contexts)
        assert any(w in blob for w in ("living room", "apartment", "vitrified", "furniture"))

    @pytest.mark.asyncio
    async def test_kitchen_query_finds_kitchen_content(self):
        from app.core.schemas import DesignRequest
        from app.services.retrieval import retrieval_service

        retrieval_service.initialize()
        req = DesignRequest(room_type="kitchen", client_preferences="modular kitchen granite")
        contexts = await retrieval_service.retrieve_for_request(req, top_k=3)
        assert any("kitchen" in c.content.lower() for c in contexts)

    def test_knowledge_base_is_wellformed(self):
        kb = get_knowledge_base()
        assert len(kb) >= 15
        for entry in kb:
            assert entry["content"].strip()
            assert entry["category"].strip()
            assert entry["source"].strip()


class TestJSONExtraction:
    @pytest.mark.parametrize("raw", [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the result:\n```json\n{"a": 1}\n```\nHope this helps!',
    ])
    def test_parses_variants(self, raw: str):
        assert extract_json(raw) == {"a": 1}

    def test_parses_arrays(self):
        assert extract_json('```json\n[1, 2, 3]\n```') == [1, 2, 3]

    def test_raises_on_garbage(self):
        with pytest.raises(ValueError):
            extract_json("no json here at all")

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            extract_json("")


class TestTruncatedJSONRepair:
    """
    Regression tests for truncated model output.

    gemma2:2b routinely exceeds the token ceiling mid-JSON. Previously the
    whole response was discarded and the agent silently fell back to an empty
    design. Recovery is what keeps the pipeline producing output.
    """

    def test_recovers_from_truncation_mid_value(self):
        assert extract_json('{"title": "Warmth", "concept": "A kitchen that welcom') == {
            "title": "Warmth"
        }

    def test_recovers_from_truncation_mid_key(self):
        out = extract_json('{"title": "X", "furniture": [{"name": "Sofa", "price"')
        assert out["title"] == "X"
        assert out["furniture"][0]["name"] == "Sofa"

    def test_recovers_from_truncation_in_array(self):
        assert extract_json('{"a": 1, "items": [1, 2, 3') == {"a": 1, "items": [1, 2]}

    def test_recovers_from_nested_truncation(self):
        out = extract_json('{"a": {"b": {"c": 1}, "d": [1,2')
        assert out["a"]["b"]["c"] == 1
        assert out["a"]["d"] == [1]

    def test_preserves_numeric_values(self):
        """
        Regression: a naive string-close turned 5000 into 5.
        Truncation recovery must never corrupt a complete value.
        """
        out = extract_json('{"title": "Perfect", "cost": 5000, "tags": ["a"], "x": "unfinis')
        assert out["cost"] == 5000
        assert out["title"] == "Perfect"
        assert out["tags"] == ["a"]

    def test_recovers_first_key_only(self):
        assert extract_json('{"title": "Incomple') == {"title": "Incomple"}

    def test_returns_object_not_nested_array(self):
        """
        Regression: the outermost-span fallback once returned ['a'] (a nested
        array) for a truncated object payload.
        """
        out = extract_json('{"title": "Perfect", "tags": ["a"], "x": "unfinis')
        assert isinstance(out, dict)
        assert out["title"] == "Perfect"


class TestEvaluationHarness:
    """
    Guards against the silent-metric failure mode.

    A benchmark case whose expectations are dropped reports a hit rate of 0.0
    and looks exactly like a retrieval regression. These tests ensure that
    kind of mistake fails loudly instead.
    """

    def test_every_golden_case_has_expectations(self):
        from app.services.evaluation import GOLDEN_DATASET

        assert GOLDEN_DATASET
        for case in GOLDEN_DATASET:
            assert case.expected_keywords, f"{case.name} has no expected keywords"

    def test_eval_case_rejects_empty_keywords(self):
        from app.core.schemas import DesignRequest
        from app.services.evaluation import EvalCase

        with pytest.raises(Exception):
            EvalCase(
                name="bad",
                request=DesignRequest(client_preferences="x"),
                expected_keywords=[],
            )

    def test_eval_case_rejects_unknown_field(self):
        from app.core.schemas import DesignRequest
        from app.services.evaluation import EvalCase

        with pytest.raises(Exception):
            EvalCase(
                name="typo",
                request=DesignRequest(client_preferences="x"),
                expected_keywrod=["sofa"],  # misspelled on purpose
            )

    @pytest.mark.asyncio
    async def test_retrieval_eval_reports_all_cases(self):
        from app.services.evaluation import GOLDEN_DATASET, evaluation_harness

        result = await evaluation_harness.evaluate_retrieval()
        assert result["cases"] == len(GOLDEN_DATASET)
        assert result["failed_cases"] == 0
        assert 0.0 < result["mean_hit_rate"] <= 1.0
        # A regression guard: retrieval must actually find things.
        assert result["mean_hit_rate"] >= 0.5

    @pytest.mark.asyncio
    async def test_evaluation_surfaces_failures(self):
        """A case that raises must be reported, not silently dropped."""
        from app.services import evaluation as ev

        class Boom:
            async def evaluate_case(self, case):
                raise RuntimeError("boom")

        original = ev.evaluation_harness.retrieval_eval
        try:
            ev.evaluation_harness.retrieval_eval = Boom()
            result = await ev.evaluation_harness.evaluate_retrieval()
            assert result["error"] == "all retrieval evaluations failed"
            assert result["failures"]
        finally:
            ev.evaluation_harness.retrieval_eval = original
