"""
Structured logging, metrics, and tracing.

Provides a consistent observability surface so every layer (API, agents,
providers) can emit machine-readable logs, Prometheus metrics, and spans
without pulling in a heavy framework.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from app.core.config import settings

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

_CONFIGURED = False


def setup_logging() -> None:
    """Configure root logging once, with optional JSON output."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    if settings.LOG_JSON:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.LOG_LEVEL)

    # Quieten noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "event", "duration_ms", "provider", "model"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


# --------------------------------------------------------------------------
# Metrics (Prometheus-style, in-process)
# --------------------------------------------------------------------------


@dataclass
class MetricValue:
    value: float = 0.0
    count: int = 0


class MetricsRegistry:
    """
    Lightweight in-process metrics registry.

    Exposes counters/histograms in Prometheus text format so the app has
    real telemetry without requiring a running Prometheus server. Swappable
    for a full client if needed.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, MetricValue] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._gauges: Dict[str, float] = {}

    @staticmethod
    def _key(name: str, labels: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a fully-qualified metric key.

        Labels are the distinguishing dimension of a real metric family: a
        single `llm.success` counter with a `provider` label is far more
        useful than a separate counter per provider.
        """
        if not labels:
            return name
        # Sorted for a deterministic key regardless of call-site ordering.
        rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{rendered}}}"

    def increment(
        self, name: str, value: float = 1.0, labels: Optional[Any] = None
    ) -> None:
        """
        Increment a counter.

        `labels` accepts either a dict or a single shorthand value (e.g.
        `labels=self.name` -> `{provider="ollama"}`), which keeps call sites
        concise at the call site.
        """
        label_map = _coerce_labels(labels)
        metric = self._counters.setdefault(self._key(name, label_map), MetricValue())
        metric.value += value
        metric.count += 1

    def gauge(self, name: str, value: float, labels: Optional[Any] = None) -> None:
        self._gauges[self._key(name, _coerce_labels(labels))] = value

    def observe(self, name: str, value: float, labels: Optional[Any] = None) -> None:
        self._histograms.setdefault(self._key(name, _coerce_labels(labels)), []).append(value)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counters": {
                k: {"value": v.value, "count": v.count}
                for k, v in self._counters.items()
            },
            "gauges": dict(self._gauges),
            "histograms": {
                k: {
                    "count": len(v),
                    "sum": sum(v),
                    "avg": (sum(v) / len(v)) if v else 0.0,
                    "p95": self._percentile(v, 0.95),
                }
                for k, v in self._histograms.items()
            },
        }

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(int(len(ordered) * pct), len(ordered) - 1)
        return ordered[idx]

    @staticmethod
    def _prom_name(key: str) -> str:
        """Sanitize a metric key into a legal Prometheus identifier."""
        base = key.split("{")[0].replace(".", "_").replace("-", "_")
        return base or "metric"

    @staticmethod
    def _prom_labels(key: str) -> str:
        """Render the label portion of a key as Prometheus label syntax."""
        if "{" not in key:
            return ""
        inner = key.split("{", 1)[1].rstrip("}")
        if not inner:
            return ""
        pairs = []
        for item in inner.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                safe_v = v.replace('"', "")
                pairs.append(f'{k.strip()}="{safe_v}"')
        return "{" + ",".join(pairs) + "}"

    def render_prometheus(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines: List[str] = []
        declared: set[str] = set()

        for key, metric in self._counters.items():
            name = self._prom_name(key)
            if name not in declared:
                lines.append(f"# TYPE {name} counter")
                declared.add(name)
            lines.append(f"{name}{self._prom_labels(key)} {metric.value}")

        for key, value in self._gauges.items():
            name = self._prom_name(key)
            if name not in declared:
                lines.append(f"# TYPE {name} gauge")
                declared.add(name)
            lines.append(f"{name}{self._prom_labels(key)} {value}")

        for key, samples in self._histograms.items():
            name = self._prom_name(key)
            if name not in declared:
                lines.append(f"# TYPE {name} summary")
                declared.add(name)
            labels = self._prom_labels(key)
            base_labels = labels[:-1] + ",quantile=\"0.95\"}" if labels else "{quantile=\"0.95\"}"
            lines.append(f"{name}_sum{labels} {sum(samples)}")
            lines.append(f"{name}_count{labels} {len(samples)}")
            lines.append(f"{name}{base_labels} {self._percentile(samples, 0.95)}")
        return "\n".join(lines) + "\n"


def _coerce_labels(labels: Optional[Any]) -> Optional[Dict[str, Any]]:
    """
    Normalise the `labels` argument.

    Accepts a dict, a bare string (treated as a generic `label`), or None,
    so call sites can pass `labels=provider` instead of building a dict.
    """
    if labels is None:
        return None
    if isinstance(labels, dict):
        return labels
    return {"label": labels}


metrics = MetricsRegistry()


# --------------------------------------------------------------------------
# Tracing
# --------------------------------------------------------------------------


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str = ""
    start_time: float = field(default_factory=time.perf_counter)
    duration_ms: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def set(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class Tracer:
    """
    Minimal distributed-tracing abstraction.

    Records spans in memory and logs them in structured form. When
    `ENABLE_TRACING` is set, spans are emitted to the log stream so they can
    be picked up by any OTLP-compatible collector.
    """

    def __init__(self) -> None:
        self.spans: List[Span] = []
        self._logger = get_logger("tracer")

    @contextmanager
    def start(self, name: str, **attributes: Any) -> Iterator[Span]:
        trace_id = uuid.uuid4().hex
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            attributes=dict(attributes),
        )
        start = time.perf_counter()
        try:
            yield span
            span.status = "ok"
        except Exception as exc:
            span.status = "error"
            span.set("error", str(exc))
            raise
        finally:
            span.duration_ms = (time.perf_counter() - start) * 1000
            if settings.ENABLE_TRACING or settings.is_production:
                self.spans.append(span)
                self._logger.info(
                    "span=%s trace=%s duration_ms=%.2f status=%s attrs=%s",
                    name,
                    trace_id,
                    span.duration_ms,
                    span.status,
                    span.attributes,
                )

    def recent(self, limit: int = 50) -> List[Span]:
        return self.spans[-limit:]


tracer = Tracer()
