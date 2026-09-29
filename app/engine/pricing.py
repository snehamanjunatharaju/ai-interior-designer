"""
Code-based pricing. All arithmetic happens here, in Python.

The original BudgetAgent asked the LLM to price a design and then tried to
"reconcile" the numbers afterwards. That is unsound: a model emitting zeros
sums to zero, and no downstream correction invents a price that was never
produced. The screenshot bug - "Rs 0 total", "items at Rs 0 but claims Rs
7.5K" - was that failure, exactly.

The division of labour is now:
  * The LLM decides *which* catalog SKUs a room needs, and why.
  * This module decides what they cost, and does the summing.

`price_design` never trusts a number from the model. Every amount is an int
read from SQLite.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from app.catalog import TIER_ESSENTIAL, catalog
from app.core.observability import get_logger, metrics

logger = get_logger(__name__)

# Soft furnishings and decor are the usual first cuts when over budget.
_DISCRETARY_CATEGORIES = {"decor", "soft", "lighting", "paint"}


def price_design(
    design: Dict[str, Any],
    budget_inr: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Price a design from the catalog and produce a closing breakdown.

    Line items carrying a valid `sku` are priced from the database; the
    model's own `estimated_price_inr` is ignored entirely. Items without a
    resolvable SKU are reported as `unpriced` rather than being given a
    guessed figure, so the total can never silently under-count.
    """
    # A layout endpoint supplies a bare list of items; a design supplies a
    # dict with a "furniture" key. Accept both so callers do not have to wrap.
    if isinstance(design, list):
        furniture = [f for f in design if isinstance(f, dict)]
    else:
        furniture = [f for f in (design.get("furniture", []) or []) if isinstance(f, dict)]

    priced_items: List[Dict[str, Any]] = []
    unpriced_items: List[Dict[str, Any]] = []
    by_category: Dict[str, int] = {}

    for entry in furniture:
        sku = str(entry.get("sku", "")).strip()
        item = catalog.get(sku) if sku else None

        if item is None:
            # Try to resolve by name as a courtesy, but never invent a price.
            name = str(entry.get("name", "")).strip().lower()
            match = next(
                (c for c in catalog.all() if c.name.lower() == name), None
            )
            if match is None:
                unpriced_items.append({
                    "name": str(entry.get("name", "unnamed item")),
                    "reason": "no catalog SKU and no exact name match",
                })
                continue
            item = match

        try:
            qty = int(float(entry.get("quantity", 1) or 1))
        except (TypeError, ValueError):
            qty = 1
        qty = max(1, qty)

        line_total = item.price_inr * qty
        priced_items.append({
            "sku": item.sku,
            "name": item.name,
            "category": item.category,
            "brand": item.brand,
            "quantity": qty,
            "unit_price_inr": item.price_inr,
            "line_total_inr": line_total,
            "tier": item.tier,
            "url": item.url,
            "dimensions_cm": {
                "width": item.width_cm,
                "depth": item.depth_cm,
                "height": item.height_cm,
            },
            # Beds can sleep more than one person per unit, so the required
            # quantity is ceil(sleepers / sleeps_per_unit) rather than the
            # sleeper count. A bunk bed satisfies a 2-bed brief at qty 1.
            "sleeps_per_unit": item.sleeps,
        })
        by_category[item.category] = by_category.get(item.category, 0) + line_total

    total = sum(i["line_total_inr"] for i in priced_items)

    breakdown: List[Dict[str, Any]] = []
    for category, amount in sorted(by_category.items(), key=lambda kv: -kv[1]):
        breakdown.append({
            "category": category,
            "amount_inr": amount,
            "percentage": round((amount / total) * 100, 1) if total else 0.0,
            "notes": "",
        })

    within_budget = True
    overage = 0
    if budget_inr:
        within_budget = total <= budget_inr
        if not within_budget:
            overage = total - budget_inr

    result: Dict[str, Any] = {
        "estimated_cost_inr": total,
        "budget_inr": budget_inr,
        "within_budget": within_budget,
        "overage_inr": overage,
        "costing_complete": not unpriced_items,
        "budget_breakdown": breakdown,
        "line_items": priced_items,
        "unpriced_items": unpriced_items,
        "cost_drivers": [
            i["name"] for i in
            sorted(priced_items, key=lambda x: -x["line_total_inr"])[:3]
        ],
        "cost_savings": [],
        "catalog_version": catalog.version(),
    }

    if unpriced_items:
        metrics.increment("pricing.unpriced_items")
        logger.warning("pricing_incomplete unpriced=%d", len(unpriced_items))

    if not within_budget:
        metrics.increment("pricing.over_budget")
        result["cost_savings"] = _suggest_cuts(priced_items, overage)

    metrics.observe("pricing.total_inr", float(total))
    return result


def required_units_for(
    category: str, requested: int, priced_items: List[Dict[str, Any]]
) -> int:
    """
    How many physical units are needed to serve `requested` people.

    A bed that sleeps one needs `requested` units. A bunk bed that sleeps two
    needs ceil(requested / 2). Without this, a two-child brief resolves to two
    bunk beds at double the price - a real money bug, not a display one.
    """
    if category != "bed":
        return requested
    sleeps_per = max(
        (i.get("sleeps_per_unit", 1) for i in priced_items
         if i.get("category") == category),
        default=1,
    )
    return math.ceil(requested / max(1, sleeps_per))


def _suggest_cuts(line_items: List[Dict[str, Any]], overage: int) -> List[str]:
    """
    Suggest which lines to drop to reach the budget, cheapest design damage
    first. Discretionary items (decor, soft furnishings) go before structural
    pieces (beds, wardrobes) because a room still functions without a rug.
    """
    ranked = sorted(
        line_items,
        key=lambda i: (
            i["category"] not in _DISCRETARY_CATEGORIES,
            i["line_total_inr"],
        ),
    )
    suggestions: List[str] = []
    running = 0
    for item in ranked:
        if running >= overage:
            break
        suggestions.append(
            f"Defer {item['name']} ({item['brand']}) - saves Rs {item['line_total_inr']:,}"
        )
        running += item["line_total_inr"]

    if running < overage:
        suggestions.append(
            f"Even deferring every discretionary item leaves Rs {overage - running:,} "
            f"to find. The full wish list costs Rs "
            f"{sum(i['line_total_inr'] for i in line_items):,}, which exceeds the budget - "
            f"phase this purchase."
        )
    return suggestions


def build_phased_plan(
    line_items: List[Dict[str, Any]],
    budget_inr: int,
) -> Dict[str, Any]:
    """
    Split a wish list into what to buy now and what to add later.

    This is the honest answer when a budget is unrealistic: "you cannot have
    all of this at Rs 10,000 - here is what Rs 10,000 actually buys." A made-up
    number that fits the budget is worse than a real number that does not,
    because the client discovers the truth at the furniture store.

    Ordering is by catalog tier, then by price: essentials first.
    """
    tier_rank = {TIER_ESSENTIAL: 0, "standard": 1, "upgrade": 2}
    ordered = sorted(
        line_items,
        key=lambda i: (tier_rank.get(i.get("tier", "standard"), 1), i["line_total_inr"]),
    )

    phase1: List[Dict[str, Any]] = []
    phase2: List[Dict[str, Any]] = []
    spent = 0

    for item in ordered:
        cost = item["line_total_inr"]
        if spent + cost <= budget_inr:
            phase1.append(item)
            spent += cost
        else:
            phase2.append(item)

    phase1_total = sum(i["line_total_inr"] for i in phase1)
    phase2_total = sum(i["line_total_inr"] for i in phase2)

    return {
        "feasible": not phase2,
        "phase1": {
            "label": "Phase 1 - buy now",
            "line_items": phase1,
            "total_inr": phase1_total,
            "remaining_inr": max(0, budget_inr - phase1_total),
        },
        "phase2": {
            "label": "Phase 2 - add later",
            "line_items": phase2,
            "total_inr": phase2_total,
        },
        "headline": (
            f"Rs {budget_inr:,} covers the essentials, not everything. "
            f"This design comes to Rs {phase1_total + phase2_total:,} in the catalog. "
            f"Here is a two-phase plan."
            if phase2 else
            f"This design fits within Rs {budget_inr:,}."
        ),
    }
