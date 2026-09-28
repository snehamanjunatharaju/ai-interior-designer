"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep tests hermetic and deterministic.
# Caching is enabled explicitly so its behaviour is actually exercised;
# tracing is off so spans do not accumulate noise across the run.
os.environ.setdefault("SEMANTIC_CACHE_ENABLED", "true")
os.environ.setdefault("ENABLE_TRACING", "false")


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def sample_request():
    """A representative design brief used across tests."""
    from app.core.schemas import DesignRequest

    return DesignRequest(
        room_type="living_room",
        client_preferences="Small compact living room for a family of 3",
        style="minimalist",
        budget="low",
        room_dimensions="12ft x 10ft",
        num_options=1,
        specific_requirements=["reading_nook"],
    )
