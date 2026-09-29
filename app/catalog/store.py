"""
Product catalog: the single source of truth for prices and dimensions.

Why this exists
---------------
The original pipeline let the LLM invent prices and then tried to "reconcile"
them in Python. That cannot work: a model that emits zeros sums to zero, and
no amount of downstream arithmetic recovers a number that was never produced.

So prices now live in SQLite. The LLM decides *which* items a room needs; the
catalog decides *what they cost* and *how big they are*. Code does the maths.

Every figure is real Indian retail pricing captured in a single reference
snapshot. Prices are indicative and dated (see `CATALOG_VERSION`) - a
production deployment would sync from live sources, but the property that
matters here is that totals are reproducible and never hallucinated.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from app.core.config import settings
from app.core.observability import get_logger, metrics

logger = get_logger(__name__)

# Bumping this forces a rebuild of the embedded database on next access.
CATALOG_VERSION = "2026-09-01"

# Cost classes used by the budget negotiator. Lower tier = buy first.
TIER_ESSENTIAL = "essential"
TIER_STANDARD = "standard"
TIER_UPGRADE = "upgrade"


class CatalogItem:
    """
    One purchasable product.

    Dimensions are in centimetres and are *required*, not optional: the layout
    engine (Phase 2) needs them to place furniture without collisions, and a
    product with unknown dimensions cannot be placed responsibly.
    """

    __slots__ = (
        "sku", "name", "category", "brand", "price_inr",
        "width_cm", "depth_cm", "height_cm", "tier", "url", "notes", "sleeps",
    )

    def __init__(
        self,
        sku: str,
        name: str,
        category: str,
        brand: str,
        price_inr: int,
        width_cm: int,
        depth_cm: int,
        height_cm: int,
        tier: str = TIER_STANDARD,
        url: str = "",
        notes: str = "",
        sleeps: int = 1,
    ) -> None:
        self.sku = sku
        self.name = name
        self.category = category
        self.brand = brand
        self.price_inr = int(price_inr)
        self.width_cm = int(width_cm)
        self.depth_cm = int(depth_cm)
        self.height_cm = int(height_cm)
        self.tier = tier
        self.url = url
        self.notes = notes
        # How many people this single unit accommodates. A bunk bed is one
        # SKU that sleeps two, so a "2 beds" brief is satisfied by quantity
        # 1 of a bunk - counting it as two units would double its price.
        self.sleeps = int(sleeps)

    def to_dict(self) -> Dict[str, object]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CatalogItem {self.sku} {self.name!r} Rs{self.price_inr}>"


class Catalog:
    """
    Read-mostly SQLite-backed catalog with an in-process cache.

    Thread-safe via a lock around writes and `check_same_thread=False` for
    reads, since FastAPI runs sync work in a threadpool.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or f"{settings.DATA_DIR}/catalog.db")
        self._lock = threading.Lock()
        self._items: Dict[str, CatalogItem] = {}
        self._version: str = ""
        self._ready = False

    # -- lifecycle ----------------------------------------------------------

    def initialize(self) -> None:
        """Create the database, seed it, and warm the in-process cache."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._migrate()
            self._seed_if_empty()
            self._load()
            self._ready = True
        logger.info("catalog_ready items=%d version=%s", len(self._items), self._version)
        metrics.gauge("catalog.items", float(len(self._items)))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS catalog_items (
                    sku        TEXT PRIMARY KEY,
                    name       TEXT NOT NULL,
                    category   TEXT NOT NULL,
                    brand      TEXT NOT NULL,
                    price_inr  INTEGER NOT NULL,
                    width_cm   INTEGER NOT NULL,
                    depth_cm   INTEGER NOT NULL,
                    height_cm  INTEGER NOT NULL,
                    tier       TEXT NOT NULL,
                    url        TEXT DEFAULT '',
                    notes      TEXT DEFAULT '',
                    sleeps     INTEGER NOT NULL DEFAULT 1,
                    version    TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_catalog_category ON catalog_items(category)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_catalog_tier ON catalog_items(tier)"
            )

    def _seed_if_empty(self) -> None:
        from app.catalog.seed import SEED_ITEMS  # local import avoids a cycle

        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM catalog_items").fetchone()
            if row["c"] > 0:
                return
            conn.executemany(
                """INSERT OR REPLACE INTO catalog_items
                   (sku, name, category, brand, price_inr, width_cm, depth_cm,
                    height_cm, tier, url, notes, sleeps, version)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        it.sku, it.name, it.category, it.brand, it.price_inr,
                        it.width_cm, it.depth_cm, it.height_cm, it.tier,
                        it.url, it.notes, it.sleeps, CATALOG_VERSION,
                    )
                    for it in SEED_ITEMS
                ],
            )
            logger.info("catalog_seeded count=%d", len(SEED_ITEMS))

    def _load(self) -> None:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM catalog_items").fetchall()
            items = {
                r["sku"]: CatalogItem(
                    r["sku"], r["name"], r["category"], r["brand"], r["price_inr"],
                    r["width_cm"], r["depth_cm"], r["height_cm"], r["tier"],
                    r["url"], r["notes"], r["sleeps"] if "sleeps" in r.keys() else 1,
                )
                for r in rows
            }
        self._items = items
        with self._connect() as conn:
            v = conn.execute("SELECT MAX(version) AS v FROM catalog_items").fetchone()
        self._version = (v["v"] if v and v["v"] else CATALOG_VERSION)

    def _ensure_ready(self) -> None:
        if not self._ready:
            self.initialize()

    # -- queries ------------------------------------------------------------

    def get(self, sku: str) -> Optional[CatalogItem]:
        self._ensure_ready()
        return self._items.get(sku)

    def all(self) -> List[CatalogItem]:
        self._ensure_ready()
        return list(self._items.values())

    def by_category(self, category: str) -> List[CatalogItem]:
        self._ensure_ready()
        return [i for i in self._items.values() if i.category == category]

    def by_tier(self, tier: str) -> List[CatalogItem]:
        self._ensure_ready()
        return [i for i in self._items.values() if i.tier == tier]

    def categories(self) -> List[str]:
        self._ensure_ready()
        return sorted({i.category for i in self._items.values()})

    def count(self) -> int:
        self._ensure_ready()
        return len(self._items)

    def version(self) -> str:
        self._ensure_ready()
        return self._version


# Module-level singleton, matching the pattern used elsewhere in the codebase.
catalog = Catalog()
