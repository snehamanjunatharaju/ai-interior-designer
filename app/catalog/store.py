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
CATALOG_VERSION = "2026-10-19"

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
        "audience",
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
        audience: str = "any",
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
        # Who this product is for: "adult", "kids" or "any". A brief that names
        # adults must never receive a kids' study table or a toy chest, and
        # this field is what makes that check possible without parsing names.
        # "any" means the product is not audience-specific (a wardrobe, a rug).
        self.audience = audience

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
        # api_name/alias -> SKU, built once at load time. Lookups by product
        # name used to be a linear scan over the whole catalog; with a name
        # index an exact (or alias) match is a dict hit.
        self._by_name: Dict[str, str] = {}
        self._alias_skus: Dict[str, str] = {}
        self._version: str = ""
        self._ready = False

    # -- lifecycle ----------------------------------------------------------

    def initialize(self) -> None:
        """Create the database, seed it, and warm the in-process cache.

        Idempotent: once the catalog is warm this returns immediately rather
        than re-running migrations, seeding, and the ready log line. Callers
        invoke it defensively before reading, so a non-guarded call must not
        turn every request into a full re-initialise.
        """
        if self._ready:
            return
        with self._lock:
            # Another thread may have finished initialising while we waited.
            if self._ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._migrate()
            self._seed_if_empty()
            self._load()
            # The seed file is authoritative. Edition changes (new SKUs, new
            # prices, withdrawn products) must reach an existing database too,
            # otherwise a stale row keeps serving an old price and a product
            # that no longer exists keeps being selected.
            self._sync_seed()
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
                    audience   TEXT NOT NULL DEFAULT 'any',
                    version    TEXT NOT NULL
                )
            """)
            # Additive migration: an existing database predates `audience`, so
            # the column is added here rather than only in the CREATE TABLE.
            columns = {
                r["name"] for r in conn.execute("PRAGMA table_info(catalog_items)")
            }
            if "audience" not in columns:
                conn.execute(
                    "ALTER TABLE catalog_items "
                    "ADD COLUMN audience TEXT NOT NULL DEFAULT 'any'"
                )
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
                                        height_cm, tier, url, notes, sleeps, audience, version)
                                      VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                                   [
                                       (
                                           it.sku, it.name, it.category, it.brand, it.price_inr,
                                           it.width_cm, it.depth_cm, it.height_cm, it.tier,
                                           it.url, it.notes, it.sleeps, it.audience, CATALOG_VERSION,
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
                    r["audience"] if "audience" in r.keys() else "any",
                )
                for r in rows
            }
        self._items = items
        by_name: Dict[str, str] = {}
        alias_skus: Dict[str, str] = {}
        for item in items.values():
            by_name.setdefault(item.name.strip().lower(), item.sku)
        # Aliases are the everyday names a client (or a model) uses for a
        # product whose catalog name is verbose: "chimney" for the
        # auto-clean hood, "countertop" for the granite slab.
        from app.catalog.seed import ITEM_ALIASES

        for alias, sku in ITEM_ALIASES.items():
            if sku in items:
                alias_skus.setdefault(alias.strip().lower(), sku)
        self._by_name = by_name
        self._alias_skus = alias_skus
        with self._connect() as conn:
            v = conn.execute("SELECT MAX(version) AS v FROM catalog_items").fetchone()
        self._version = (v["v"] if v and v["v"] else CATALOG_VERSION)

    def _sync_seed(self) -> None:
        """Bring an existing database in line with the seed edition.

        Idempotent and cheap: it only writes when a row differs, and it
        deletes rows the seed no longer defines so a withdrawn product cannot
        be selected and priced.
        """
        from app.catalog.seed import SEED_ITEMS  # local import avoids a cycle

        with self._connect() as conn:
            existing = {
                r["sku"]: r for r in conn.execute("SELECT * FROM catalog_items")
            }
            seed_skus = {it.sku for it in SEED_ITEMS}
            stale = [sku for sku in existing if sku not in seed_skus]
            if stale:
                conn.executemany(
                    "DELETE FROM catalog_items WHERE sku = ?",
                    [(sku,) for sku in stale],
                )
                logger.info("catalog_removed_withdrawn count=%d", len(stale))
            for it in SEED_ITEMS:
                row = existing.get(it.sku)
                if (
                    row is not None
                    and row["price_inr"] == it.price_inr
                    and row["name"] == it.name
                    and row["category"] == it.category
                    and row["brand"] == it.brand
                ):
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO catalog_items
                       (sku, name, category, brand, price_inr, width_cm, depth_cm,
                        height_cm, tier, url, notes, sleeps, audience, version)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        it.sku, it.name, it.category, it.brand, it.price_inr,
                        it.width_cm, it.depth_cm, it.height_cm, it.tier,
                        it.url, it.notes, it.sleeps, it.audience, CATALOG_VERSION,
                    ),
                )

    def _ensure_ready(self) -> None:
        if not self._ready:
            self.initialize()

    # -- queries ------------------------------------------------------------

    def get(self, sku: str) -> Optional[CatalogItem]:
        self._ensure_ready()
        # SKUs are stored uppercase; a model echoing one back in lower case is
        # still naming the same product, and failing to find it used to send
        # the line down the "invented SKU" repair path.
        key = (sku or "").strip()
        return self._items.get(key) or self._items.get(key.upper())

    def resolve_sku(self, sku: Optional[str] = None, name: Optional[str] = None) -> Optional[str]:
        """Find the SKU for a product by exact SKU, exact name, or alias.

        Returns None rather than guessing: a near-match on a product name is
        how a countertop gets replaced with a kitchen sink.
        """
        self._ensure_ready()
        key = (sku or "").strip()
        if key:
            item = self._items.get(key) or self._items.get(key.upper())
            if item is not None:
                return item.sku
        label = (name or "").strip().lower()
        if not label:
            return None
        return self._by_name.get(label) or self._alias_skus.get(label)

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

    def for_room(self, room_type: str) -> List[CatalogItem]:
        """Products that belong in a room of this type.

        A category with no room restriction (a lamp, a rug) is available
        everywhere; a bed is not available in a kitchen. The restriction lives
        in `seed.CATEGORY_ROOMS`.
        """
        from app.catalog.seed import CATEGORY_ROOMS

        self._ensure_ready()
        room = (room_type or "").strip().lower()
        if not room:
            return self.all()
        return [
            i for i in self._items.values()
            if not CATEGORY_ROOMS.get(i.category)
            or room in CATEGORY_ROOMS[i.category]
        ]

    def allowed_in_room(self, item: CatalogItem, room_type: str) -> bool:
        """Is this product appropriate for the given room type?"""
        from app.catalog.seed import CATEGORY_ROOMS

        room = (room_type or "").strip().lower()
        if not room:
            return True
        rooms = CATEGORY_ROOMS.get(item.category)
        return not rooms or room in rooms

    def count(self) -> int:
        self._ensure_ready()
        return len(self._items)

    def version(self) -> str:
        self._ensure_ready()
        return self._version


# Module-level singleton, matching the pattern used elsewhere in the codebase.
catalog = Catalog()
