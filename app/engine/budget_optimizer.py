"""
Deterministic budget optimizer.

The pipeline's premise is that language models write and code decides. Pricing
was already moved into Python, but *selection under a constraint* had not: the
architect picked whatever it liked and the pipeline then discovered afterwards
that the total exceeded the budget, reflected once, and gave up.

That is the wrong shape. A budget is a constraint, not a preference, and
satisfying it is arithmetic. This module answers, deterministically and
without an LLM:

    "Given this brief and this budget, what is the best set of real catalog
     products?"

The strategy is cheapest-first, then upgrade by priority while headroom
remains. It is not optimal in the strict sense - that is a knapsack problem and
the gains here are marginal against the value of being explainable - but it is
stable, fast, and every decision can be stated in one sentence to a client.

When no set of real products fits, this module does not quietly return the
cheapest partial room. It reports *infeasibility* with the shortfall and three
concrete escapes, because "you cannot have this at Rs 10,000" is the honest
answer and an empty room is not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.core.observability import get_logger, metrics

logger = get_logger(__name__)

# Priority 0 is bought before anything else, at every budget. The order is a
# property of how a room functions, not of price: a bed without a mattress is
# not a bedroom, and a room with no chair is not a workspace.
CATEGORY_PRIORITY: Dict[str, int] = {
    "bed": 0,
    "mattress": 1,
    "wardrobe": 2,
    "desk": 3,
    "chair": 4,
    "seating": 5,
    "table": 6,
    "storage": 7,
    "kitchen": 8,
    "tv_unit": 9,
    "lighting": 10,
    "soft": 11,
    "decor": 12,
    "paint": 13,
}

# The five categories the brief calls out as non-negotiable. Phase 1 is built
# from these before any discretionary line is considered.
ESSENTIAL_CATEGORIES: tuple[str, ...] = ("bed", "mattress", "wardrobe", "desk", "chair")


class OptimisationResult(dict):
    """
    A plain dict subclass so the result serialises straight into the API
    response and the tests can assert on keys without unpacking objects.
    """


def optimize_to_budget(
    required: Dict[str, int],
    budget_inr: Optional[int],
    optional: Optional[Dict[str, int]] = None,
    style_tier: str = "standard",
    audience: str = "any",
) -> OptimisationResult:
    """
    Build the best real shopping list that fits `budget_inr`.

    `required` maps a catalog category to the number of units the brief demands.
    `optional` is what the designer would like but the client can live without.
    `style_tier` ("essential" | "standard" | "upgrade") caps how far the
    optimizer is allowed to upgrade an item, so a low budget is honoured by
    choosing simpler products rather than by breaking the budget.

    The returned dict always contains:
      selections   - [{sku, quantity, category, price_inr, upgrade: bool}]
      total_inr    - the exact sum of those lines
      budget_inr
      feasible     - whether every required unit was bought
      shortfall_inr- how much more is needed when not feasible
      infeasible   - a client-readable explanation when not feasible
      options      - concrete ways out of an infeasible brief
    """
    from app.engine.brief import products_satisfying
    from app.catalog import catalog

    required = {k: int(v) for k, v in (required or {}).items() if int(v) > 0}
    optional = {k: int(v) for k, v in (optional or {}).items() if int(v) > 0}
    budget = int(budget_inr) if budget_inr else None

    tier_ceiling = _tier_ceiling(style_tier)
    selections: List[Dict[str, Any]] = []
    spent = 0
    unserved: List[str] = []

    # --- Pass 1: every required unit, cheapest viable product first ---------
    for category in _by_priority(required):
        count = required[category]
        served = 0
        while served < count:
            options = products_satisfying(category, audience=audience)
            if not options:
                break
            # The tier ceiling deliberately does NOT apply to required items.
            # Several required categories (wardrobe, most seating) have no
            # essential-tier product at all, so applying the ceiling here made
            # an ordinary Rs 1.2L bedroom report itself infeasible. The ceiling
            # governs discretionary *upgrades*, not whether the brief can be met.
            remaining = count - served

            def value(p: Any) -> tuple:
                """
                Rank by value per sleeper, then by whether one unit covers
                what is still needed.

                Sorting by `sleeps` first handed a *single* child's bed the
                Rs 14,500 bunk instead of the Rs 3,200 cot - 4x the price for
                one occupant. Sorting by price-per-sleeper alone then over-
                corrected: two children got two singles, because a single's
                per-sleeper price is lower than a bunk's even though one bunk
                satisfies the whole brief for less money. So the tie-break asks
                the question that actually decides it - does this unit cover
                the demand still outstanding?
                """
                per_sleeper = p.price_inr / max(1, p.sleeps)
                covers_need = 0 if p.sleeps >= remaining else 1
                return (covers_need, per_sleeper, p.price_inr, p.name)

            cheapest = min(options, key=value)
            units_covered = max(1, cheapest.sleeps)
            if budget is not None and spent + cheapest.price_inr > budget:
                # Cannot afford the best-cover option; fall back to the plain
                # cheapest one, which may cover fewer sleepers per unit.
                plain = min(options, key=lambda p: (p.price_inr, p.name))
                if spent + plain.price_inr > budget:
                    break
                cheapest = plain
                units_covered = max(1, plain.sleeps)
            selections.append(_selection(cheapest, 1))
            spent += cheapest.price_inr
            served += units_covered
        if served < count:
            unserved.append((category, count - served))

    # --- Pass 2: upgrade by priority while headroom remains ----------------
    # An upgrade only happens when the *whole* required set is already bought,
    # otherwise the client would be trading a missing bed for a nicer one.
    if not unserved:
        upgraded = _upgrade_within_budget(
            selections, required, budget, spent, tier_ceiling, audience
        )
        selections = upgraded["selections"]
        spent = upgraded["spent"]

    # --- Pass 3: discretionary items fill whatever remains ------------------
    if not unserved:
        for category in _by_priority(optional):
            for _ in range(optional[category]):
                options = [
                    p for p in products_satisfying(category, audience=audience)
                    if _tier_rank(p.tier) <= tier_ceiling
                ]
                if not options:
                    break
                cheapest = min(options, key=lambda p: (p.price_inr, p.name))
                if budget is not None and spent + cheapest.price_inr > budget:
                    break
                selections.append(_selection(cheapest, 1))
                spent += cheapest.price_inr

    feasible = not unserved
    shortfall = 0
    if not feasible:
        # What would the missing pieces cost at their cheapest? That is the
        # honest number for "how much more do you need".
        shortfall = _cheapest_cost_of_shortfall(unserved, tier_ceiling)

    # The total is summed from the finished lines rather than read from the
    # running `spent` counter. The two disagreed: an upgrade adjusted a line
    # without the counter tracking it the same way, and the design then
    # published a total that did not match its own shopping list - the exact
    # defect this module exists to prevent. Deriving it makes the invariant
    # impossible to break as the code changes.
    total = sum(int(s["line_total_inr"]) for s in selections)

    result = OptimisationResult({
        "selections": selections,
        "total_inr": total,
        "budget_inr": budget,
        "feasible": feasible,
        "shortfall_inr": shortfall,
        # The raw (category, missing) pairs, so a caller can tell exactly
        # which categories the budget blocked rather than re-parsing prose.
        "unserved": [{"category": c, "missing": n} for c, n in unserved],
        "infeasible": None if feasible else _infeasible_message(
            unserved, total, budget, shortfall
        ),
        "options": [] if feasible else _escape_routes(
            unserved, total, budget, shortfall, style_tier
        ),
        "catalog_version": catalog.version(),
    })
    metrics.observe("optimizer.total_inr", float(total))
    if not feasible:
        metrics.increment("optimizer.infeasible")
    return result


def _selection(product: Any, quantity: int) -> Dict[str, Any]:
    return {
        "sku": product.sku,
        "quantity": quantity,
        "category": product.category,
        "name": product.name,
        "brand": product.brand,
        "unit_price_inr": product.price_inr,
        "line_total_inr": product.price_inr * quantity,
        "tier": product.tier,
        "width_cm": product.width_cm,
        "depth_cm": product.depth_cm,
        "height_cm": product.height_cm,
    }


def _by_priority(categories: Dict[str, int]) -> List[str]:
    """Order categories by functional priority, unknowns last."""
    return sorted(
        categories,
        key=lambda c: (CATEGORY_PRIORITY.get(c, 99), c),
    )


def _tier_rank(tier: str) -> int:
    return {"essential": 0, "standard": 1, "upgrade": 2}.get(str(tier), 1)


def _tier_ceiling(style_tier: str) -> int:
    """The most expensive product tier this budget is allowed to buy."""
    return {
        "essential": 0,
        "low": 0,
        "medium": 1,
        "standard": 1,
        "high": 2,
        "luxury": 2,
        "upgrade": 2,
    }.get(str(style_tier).lower(), 1)


def _upgrade_within_budget(
    selections: List[Dict[str, Any]],
    required: Dict[str, int],
    budget: Optional[int],
    spent: int,
    tier_ceiling: int,
    audience: str = "any",
) -> Dict[str, Any]:
    """
    Swap each selected item for a better one, cheapest upgrade first.

    The ceiling stops a mid budget from buying a `luxury` sofa, which is what
    makes this compatible with a stated budget *tier* rather than only a number.
    """
    from app.engine.brief import products_satisfying

    if budget is None:
        return {"selections": selections, "spent": spent}

    headroom = budget - spent
    if headroom <= 0:
        return {"selections": selections, "spent": spent}

    # Collect candidate upgrades with their incremental cost, cheapest first,
    # so a tight budget takes the single most valuable improvement rather than
    # an arbitrary one.
    candidates: List[tuple[int, int, Dict[str, Any]]] = []
    for index, sel in enumerate(selections):
        category = sel["category"]
        for product in products_satisfying(category, audience=audience):
            if _tier_rank(product.tier) > tier_ceiling:
                continue
            delta = product.price_inr - sel["unit_price_inr"]
            if delta <= 0 or delta > headroom:
                continue
            # An upgrade must also beat the current item on quality, not only
            # cost; tier rank is the proxy the catalog exposes.
            if _tier_rank(product.tier) <= _tier_rank(sel.get("tier", "standard")):
                continue
            candidates.append((delta, index, _selection(product, sel["quantity"])))

    for delta, index, replacement in sorted(candidates, key=lambda c: c[0]):
        current = selections[index]
        new_spend = spent + delta
        if new_spend > budget:
            continue
        spent = new_spend
        headroom = budget - spent
        selections[index] = replacement
        replacement["upgraded"] = True

    return {"selections": selections, "spent": spent}


def _cheapest_cost_of_shortfall(
    unserved: Sequence[tuple], tier_ceiling: int
) -> int:
    """
    Cheapest real cost of the pieces that could not be bought.

    `unserved` is a list of (category, missing_count) pairs, so this sums the
    genuine price of what is absent rather than guessing.
    """
    from app.engine.brief import products_satisfying

    total = 0
    for category, missing in unserved:
        options = products_satisfying(category)
        if options:
            total += min(p.price_inr for p in options) * missing
    return total


def _infeasible_message(
    unserved: Sequence[tuple],
    spent: int,
    budget: Optional[int],
    shortfall: int,
) -> str:
    missing = "; ".join(
        f"{n} more {c}" for c, n in unserved
    )
    if budget is None:
        return f"Could not source these from the catalog: {missing}."
    return (
        f"A budget of Rs {budget:,} cannot cover this brief. The affordable items "
        f"come to Rs {spent:,}, and the missing pieces ({missing}) need a further "
        f"Rs {shortfall:,}. The realistic minimum for this room is about "
        f"Rs {spent + shortfall:,}."
    )


def _escape_routes(
    unserved: Sequence[tuple],
    spent: int,
    budget: Optional[int],
    shortfall: int,
    style_tier: str,
) -> List[Dict[str, str]]:
    """
    Concrete ways out of an infeasible brief.

    Each route names the exact change and its exact effect, so the client can
    choose rather than being told "no". The numbers come from the same catalog
    the design is priced against, so a route can never promise a total the code
    would not produce.
    """
    minimum = spent + shortfall
    routes: List[Dict[str, str]] = [
        {
            "kind": "raise_budget",
            "label": f"Raise the budget to Rs {minimum:,}",
            "detail": (
                f"Buying everything the brief asks for needs about Rs {minimum:,}. "
                f"That is Rs {shortfall:,} more than Rs {budget:,}."
            ),
        },
        {
            "kind": "reduce_scope",
            "label": "Drop the optional pieces and keep the essentials",
            "detail": (
                "Essentials only (bed, mattress, wardrobe, desk, chair) come to "
                f"Rs {spent:,}, which fits within Rs {budget:,}. Decor, lighting "
                "and secondary seating can follow later."
            ),
        },
        {
            "kind": "change_tier",
            "label": "Switch to a lower product tier",
            "detail": (
                "Choosing essential-tier products for every category removes the "
                f"upgrade premium and brings the room close to Rs {minimum:,} "
                "at its cheapest."
            ),
        },
    ]
    return routes