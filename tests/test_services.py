"""
Tests for the LLM layer, semantic cache, and agent logic.

LLM calls are mocked, so the suite is fast, deterministic, and offline.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

import numpy as np
import pytest

from app.services.llm_provider import CircuitBreaker, LLMProvider
from app.services.semantic_cache import SemanticCache


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class FlakyProvider(LLMProvider):
    """Provider that fails a configurable number of times before succeeding."""

    name = "flaky"

    def __init__(self, fail_times: int = 1) -> None:
        super().__init__()
        self.calls = 0
        self.fail_times = fail_times

    async def _complete(self, prompt, system, **kwargs) -> tuple[str, int, int]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated failure")
        return "ok", 10, 5


class JSONProvider(LLMProvider):
    """Provider returning a fixed JSON payload."""

    name = "jsonprov"

    def __init__(self, payload: str) -> None:
        super().__init__()
        self.payload = payload

    async def _complete(self, prompt, system, **kwargs) -> tuple[str, int, int]:
        return self.payload, 10, 20


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        cb = CircuitBreaker(threshold=3, recovery_timeout=60)
        for _ in range(3):
            await cb.record_failure()
        assert cb.state == "open"
        assert await cb.allow() is False

    @pytest.mark.asyncio
    async def test_success_resets_failures(self):
        cb = CircuitBreaker(threshold=3, recovery_timeout=60)
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_success()
        assert cb.failures == 0
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_half_opens_after_timeout(self):
        cb = CircuitBreaker(threshold=1, recovery_timeout=0.05)
        await cb.record_failure()
        assert cb.state == "open"
        await asyncio.sleep(0.06)
        assert await cb.allow() is True
        assert cb.state == "half_open"

    @pytest.mark.asyncio
    async def test_half_open_success_closes(self):
        cb = CircuitBreaker(threshold=1, recovery_timeout=0.01)
        await cb.record_failure()
        await asyncio.sleep(0.02)
        await cb.allow()
        await cb.record_success()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(threshold=1, recovery_timeout=0.01)
        await cb.record_failure()
        await asyncio.sleep(0.02)
        await cb.allow()
        await cb.record_failure()
        assert cb.state == "open"


class TestProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_shape(self):
        result = await FlakyProvider(fail_times=0).complete("hi")
        assert result["text"] == "ok"
        assert result["provider"] == "flaky"
        assert result["prompt_tokens"] == 10
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_failure_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            await FlakyProvider(fail_times=99).complete("hi")

    @pytest.mark.asyncio
    async def test_recovery_after_transient_failure(self):
        p = FlakyProvider(fail_times=1)
        with pytest.raises(RuntimeError):
            await p.complete("hi")
        result = await p.complete("hi")
        assert result["text"] == "ok"

    @pytest.mark.asyncio
    async def test_open_circuit_fails_fast(self):
        p = FlakyProvider(fail_times=99)
        p.breaker.threshold = 1
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await p.complete("hi")
        assert p.breaker.state == "open"
        calls_before = p.calls
        with pytest.raises(RuntimeError, match="Circuit open"):
            await p.complete("hi")
        # The key property: no additional backend call was made.
        assert p.calls == calls_before

    def test_cost_estimation_is_zero_for_local(self):
        class Local(FlakyProvider):
            name = "ollama"

        assert Local(0).estimate_cost_inr(1_000_000, 1_000_000) == 0.0


# --------------------------------------------------------------------------
# Semantic cache
# --------------------------------------------------------------------------


class TestSemanticCache:
    @pytest.fixture
    def cache(self) -> SemanticCache:
        # High threshold, matching the production default.
        return SemanticCache(dim=32, threshold=0.9, ttl_seconds=60, max_entries=3)

    @pytest.fixture
    def vec(self):
        from app.services.embeddings import HashingEmbedder

        return HashingEmbedder(dim=32)

    def test_exact_hit(self, cache, vec):
        payload = {"style": "modern"}
        v = vec.embed("modern living room")
        cache.set(payload, v, {"result": 1})
        assert cache.get(payload, v) == {"result": 1}

    def test_semantic_hit_on_close_paraphrase(self, cache, vec):
        """
        Near-identical phrasing should hit.

        The threshold is intentionally strict, so this asserts the safe
        property: a genuinely equivalent request is served from cache, while
        a merely similar one is not (see the contrast test below).
        """
        a, b = {"text": "small living room"}, {"text": "small living room"}
        va = vec.embed("small living room with a sofa")
        vb = vec.embed("small living room with a sofa")
        cache.set(a, va, {"cached": True})
        assert cache.get(b, vb) == {"cached": True}

    def test_contradictory_request_is_not_cached(self, cache, vec):
        """
        Safety-critical: opposite intent must never collide.

        'tiny' and 'spacious' are antonyms but share surface tokens, so a
        low threshold would serve the wrong design. They must not hit.
        """
        cache.set({"text": "tiny living room"}, vec.embed("tiny living room"), {"wrong": True})
        assert cache.get(
            {"text": "spacious living room"}, vec.embed("spacious living room")
        ) is None

    def test_miss_on_unrelated(self, cache, vec):
        a, b = {"text": "living room"}, {"text": "kitchen ventilation"}
        cache.set(a, vec.embed("living room"), {"x": 1})
        assert cache.get(b, vec.embed("kitchen ventilation chimney")) is None

    def test_eviction_respects_max_entries(self, cache, vec):
        for i in range(5):
            cache.set({"i": i}, vec.embed(f"query number {i} distinct"), {"v": i})
        assert len(cache._entries) == 3

    def test_ttl_expiry(self, vec):
        cache = SemanticCache(dim=32, threshold=0.9, ttl_seconds=1, max_entries=10)
        cache.set({"a": 1}, vec.embed("query text"), {"v": 1})
        time.sleep(1.1)
        assert cache.get({"a": 1}, vec.embed("query text")) is None

    def test_hit_refreshes_ttl(self, vec):
        cache = SemanticCache(dim=32, threshold=0.9, ttl_seconds=1, max_entries=10)
        v = vec.embed("stable query string")
        cache.set({"a": 1}, v, {"v": 1})
        for _ in range(3):
            time.sleep(0.4)
            assert cache.get({"a": 1}, v) == {"v": 1}

    def test_stats(self, cache, vec):
        v = vec.embed("something")
        cache.set({"a": 1}, v, {"v": 1})
        cache.get({"a": 1}, v)
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["size"] == 1
        assert 0.0 <= stats["hit_rate"] <= 1.0

    def test_key_is_order_insensitive(self):
        assert SemanticCache.make_key({"a": 1, "b": 2}) == SemanticCache.make_key(
            {"b": 2, "a": 1}
        )


# --------------------------------------------------------------------------
# Agent logic (no LLM)
# --------------------------------------------------------------------------


class TestBudgetAgent:
    @pytest.fixture
    def agent(self):
        from app.agents.orchestrator import BudgetAgent

        return BudgetAgent()

    def test_reconciles_arithmetic_from_breakdown(self, agent, sample_request):
        """LLM headline total is ignored; the parts are trusted."""
        costing = {
            "estimated_cost_inr": 999999,
            "budget_breakdown": [
                {"category": "Furniture", "amount_inr": 50000, "percentage": 999},
                {"category": "Lighting", "amount_inr": 30000, "percentage": 999},
            ],
        }
        out = agent._reconcile(costing, {"furniture": []}, sample_request, 100_000)
        assert out["estimated_cost_inr"] == 80_000
        assert sum(b["percentage"] for b in out["budget_breakdown"]) == pytest.approx(100, abs=0.5)
        assert out["within_budget"] is True

    def test_flags_over_budget(self, agent, sample_request):
        costing = {
            "budget_breakdown": [{"category": "Furniture", "amount_inr": 500_000}],
        }
        out = agent._reconcile(costing, {"furniture": []}, sample_request, 100_000)
        assert out["within_budget"] is False
        assert any("exceeds" in s.lower() for s in out["cost_savings"])

    def test_handles_negative_amounts(self, agent, sample_request):
        costing = {"budget_breakdown": [{"category": "X", "amount_inr": -500}]}
        out = agent._reconcile(costing, {"furniture": []}, sample_request, 100_000)
        assert all(b["amount_inr"] >= 0 for b in out["budget_breakdown"])

    def test_derives_from_furniture_when_breakdown_missing(self, agent, sample_request):
        design = {
            "furniture": [
                {"name": "Sofa", "estimated_price_inr": 20000, "quantity": 1},
                {"name": "Table", "estimated_price_inr": 5000, "quantity": 2},
            ]
        }
        out = agent._reconcile({}, design, sample_request, 1_000_000)
        # 20000 + (5000 * 2)
        assert out["estimated_cost_inr"] == 30_000


class TestCriticAgent:
    @pytest.fixture
    def agent(self):
        from app.agents.orchestrator import CriticAgent

        return CriticAgent()

    def test_clamps_out_of_range_scores(self, agent):
        out = agent._sanitize({
            "scores": {"coherence": 99, "feasibility": -5, "budget_adherence": "bad"},
            "overall": 500,
        })
        assert out["scores"]["coherence"] == 10.0
        assert out["scores"]["feasibility"] == 0.0
        assert out["overall"] == 10.0

    def test_auto_triggers_revision_on_low_score(self, agent):
        out = agent._sanitize({"scores": {"aesthetics": 3}, "overall": 4})
        assert out["needs_revision"] is True

    def test_no_revision_on_high_score(self, agent):
        out = agent._sanitize({"scores": {"aesthetics": 9}, "overall": 8.5})
        assert out["needs_revision"] is False

    def test_computes_overall_when_missing(self, agent):
        out = agent._sanitize({
            "scores": {"coherence": 8, "feasibility": 8, "budget_adherence": 8,
                       "aesthetics": 8, "specificity": 8}
        })
        assert out["overall"] == pytest.approx(8.0)

    def test_defaults_when_scores_absent(self, agent):
        out = agent._sanitize({})
        assert all(0 <= v <= 10 for v in out["scores"].values())


class TestDesignPipelineHelpers:
    @pytest.fixture
    def pipeline(self):
        from app.agents.orchestrator import DesignPipeline

        return DesignPipeline()

    def test_to_option_handles_garbage(self, pipeline, sample_request):
        """Malformed LLM output must never crash the response builder."""
        design = {
            "title": 123,
            "concept": None,
            "furniture": [
                "not a dict",
                {"name": "Sofa", "estimated_price_inr": "not a number", "quantity": "bad"},
            ],
            "color_palette": "not a dict",
            "zones": ["ok", None, 42],
        }
        option = pipeline._to_option(design, sample_request)
        assert option.title  # coerced to string, non-empty
        assert len(option.furniture) == 1
        assert option.furniture[0].estimated_price_inr == 0
        assert option.furniture[0].quantity == 1

    def test_to_option_estimates_cost_from_furniture(self, pipeline, sample_request):
        design = {
            "title": "T",
            "furniture": [{"name": "Sofa", "estimated_price_inr": 10000, "quantity": 2}],
        }
        assert pipeline._to_option(design, sample_request).estimated_cost_inr == 20_000

    def test_to_option_empty_design(self, pipeline, sample_request):
        option = pipeline._to_option({}, sample_request)
        assert option.title
        assert option.furniture == []
