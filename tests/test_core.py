"""
Tests for the core primitives: config, schemas, security, observability.

These are pure and fast, with no LLM or network dependency.
"""

from __future__ import annotations

import time

import pytest

from app.core.config import Settings
from app.core.observability import MetricsRegistry
from app.core.schemas import DesignOption, DesignRequest
from app.core.security import RateLimiter, verify_api_key


class TestSchemas:
    def test_request_defaults(self, sample_request: DesignRequest):
        assert sample_request.room_type.value == "living_room"
        assert sample_request.budget.value == "low"
        assert sample_request.num_options == 1

    def test_deduplicates_list_fields(self):
        req = DesignRequest(
            client_preferences="x",
            furniture=["Sofa", "sofa", "  Chair  ", ""],
        )
        assert req.furniture == ["Sofa", "Chair"]

    @pytest.mark.parametrize("bad", [0, 9, -1])
    def test_rejects_out_of_range_num_options(self, bad: int):
        with pytest.raises(Exception):
            DesignRequest(client_preferences="x", num_options=bad)

    def test_rejects_empty_brief(self):
        with pytest.raises(Exception):
            DesignRequest(client_preferences="")

    def test_design_option_tolerates_missing_fields(self):
        """A sparse LLM payload must still coerce into the schema."""
        option = DesignOption(title="Minimal Room")
        assert option.furniture == []
        assert option.estimated_cost_inr == 0


class TestConfig:
    def test_production_detection(self):
        assert Settings(ENVIRONMENT="production").is_production
        assert not Settings(ENVIRONMENT="development").is_production

    def test_pricing_has_ollama_free(self):
        assert Settings().pricing_inr["ollama"] == 0.0

    def test_no_secret_in_defaults(self):
        """Secrets must never have a baked-in default."""
        s = Settings()
        assert s.API_KEY in (None, "")
        assert s.OPENAI_API_KEY in (None, "")
        assert s.ANTHROPIC_API_KEY in (None, "")


class TestSecurity:
    def test_auth_disabled_without_key(self):
        assert verify_api_key(None) is True
        assert verify_api_key("anything") is True

    def test_auth_enabled_with_key(self, monkeypatch):
        monkeypatch.setattr("app.core.security.settings.API_KEY", "secret123", raising=False)
        assert verify_api_key("secret123") is True
        assert verify_api_key("wrong") is False
        assert verify_api_key(None) is False

    def test_rate_limiter_allows_within_window(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.check("ip")[0] is True
        assert limiter.check("ip")[0] is True
        assert limiter.check("ip")[0] is True
        assert limiter.check("ip")[0] is False

    def test_rate_limiter_isolates_keys(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.check("a")[0] is True
        assert limiter.check("b")[0] is True  # different client unaffected

    def test_rate_limiter_recovers_after_window(self):
        limiter = RateLimiter(max_requests=1, window_seconds=1)
        assert limiter.check("ip")[0] is True
        assert limiter.check("ip")[0] is False
        time.sleep(1.1)
        assert limiter.check("ip")[0] is True


class TestMetrics:
    def test_counter_with_labels(self):
        m = MetricsRegistry()
        m.increment("llm.success", labels="ollama")
        m.increment("llm.success", labels="ollama")
        m.increment("llm.success", labels="openai")
        snap = m.snapshot()["counters"]
        assert snap["llm.success{label=ollama}"]["value"] == 2
        assert snap["llm.success{label=openai}"]["value"] == 1

    def test_label_ordering_is_deterministic(self):
        m = MetricsRegistry()
        m.increment("x", labels={"b": "2", "a": "1"})
        m.increment("x", labels={"a": "1", "b": "2"})
        assert len(m.snapshot()["counters"]) == 1

    def test_gauge_overwrites(self):
        m = MetricsRegistry()
        m.gauge("g", 1.0)
        m.gauge("g", 5.0)
        assert m.snapshot()["gauges"]["g"] == 5.0

    def test_histogram_stats(self):
        m = MetricsRegistry()
        for v in (1, 2, 3, 4, 5):
            m.observe("h", float(v))
        stats = m.snapshot()["histograms"]["h"]
        assert stats["count"] == 5
        assert stats["sum"] == 15
        assert stats["avg"] == 3.0

    def test_prometheus_render(self):
        m = MetricsRegistry()
        m.increment("http.requests", labels="GET /health")
        m.observe("http.latency_ms", 12.0)
        text = m.render_prometheus()
        assert "# TYPE http_requests counter" in text
        assert 'http_requests{label="GET /health"}' in text
        assert "http_latency_ms_count" in text
