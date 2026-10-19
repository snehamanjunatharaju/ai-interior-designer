"""
Making sure the N options are actually N different options.

The pipeline asked the model for three designs and got three variations on one
idea: same layout, same palette, same budget strategy, with the title swapped.
From the client's side that is one option presented three times, and the
"2 alternatives" button is a lie.

Two things cause it. A single architect pass produces one design, and asking a
2B model for "two more different options" reliably returns near-copies. And
nothing measured the result, so near-copies were served without comment.

This module does the measuring. Options are reduced to a comparable fingerprint
- layout, palette, product mix, budget strategy - and two options that are too
similar are rejected rather than shown. `select_distinct` is the entry point the
orchestrator calls, and it is deliberately deterministic: no embedding model, no
LLM judgement, so the same brief always produces the same set of options and the
behaviour can be asserted in a test.

Similarity is measured on structured features rather than prose. Two options
with different words but the same six products in the same corners are the
same option, and only the structure shows that.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Titles that differ only in an ordinal are the same option.
_ORDINAL = re.compile(r"\b(option|variant|design|idea)\s*[#\-]?\s*[0-9ivx]+\b", re.I)
_FILLER = re.compile(
    r"\b(the|a|an|with|and|for|of|in|on|to|modern|contemporary|minimal|"
    r"elegant|style|design|option|space|room|interior)\b",
    re.I,
)

# A pair of options more similar than this is the same option twice. Tuned by
# hand against the real generated set: genuinely different options score around
# 0.3-0.5, near-copies land above 0.75.
DUPLICATE_THRESHOLD = 0.72


def fingerprint(option: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce an option to the features that make it a distinct design.

    Prose is deliberately excluded from the comparison. Two designs can be
    described in completely different words and still be the same room, and
    that is precisely the case worth catching.
    """
    furniture = [
        f for f in (option.get("furniture") or []) if isinstance(f, dict)
    ]

    # The product mix, ignoring quantity: two rooms with one bed each are the
    # same room as far as "is this a different option" goes.
    skus = frozenset(str(f.get("sku", "")) for f in furniture if f.get("sku"))

    categories = frozenset(
        str(f.get("category", "")) for f in furniture if f.get("category")
    )

    palette = option.get("color_palette") or {}
    hex_codes = tuple(sorted(
        str(h).upper() for h in (palette.get("hex_codes") or []) if h
    ))

    zones = frozenset(
        _normalise_zone(z) for z in (option.get("zones") or []) if z
    )

    total = option.get("estimated_cost_inr") or 0

    return {
        "skus": skus,
        "categories": categories,
        "hex_codes": hex_codes,
        "zones": zones,
        "budget_band": _budget_band(total),
        "title_core": _title_core(str(option.get("title", ""))),
    }


def _normalise_zone(zone: Any) -> str:
    """Zones compare by meaning, not by wording ("study zone" == "study")."""
    if isinstance(zone, dict):
        zone = zone.get("name") or zone.get("purpose") or ""
    words = [w for w in re.split(r"\W+", str(zone).lower()) if w]
    meaningful = [w for w in words if not _FILLER.fullmatch(w)]
    return " ".join(meaningful or words[:1])


def _title_core(title: str) -> str:
    """Title with ordinals and filler removed, so 'Option A' and 'Option B' match."""
    cleaned = _ORDINAL.sub(" ", title)
    words = [w for w in re.split(r"\W+", cleaned.lower()) if w]
    meaningful = [w for w in words if not _FILLER.fullmatch(w)]
    return " ".join(meaningful)


def _budget_band(total: int) -> str:
    """
    Coarse budget strategy.

    Compared as a band rather than an exact figure: two options differing by a
    few hundred rupees are the same strategy, and demanding an exact match
    would reject sensible variations of one plan.
    """
    try:
        value = int(total or 0)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        return "unknown"
    magnitude = len(str(value))
    return f"~{magnitude}"


def similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """
    Weighted similarity in [0, 1] between two fingerprints.

    The product mix dominates, because the shopping list *is* the design. A
    shared palette or title counts for much less, since those are decoration on
    top of the same purchase.

    Weighted rather than averaged because the features are not equally
    informative: differing in six of eight SKUs is a different room even if
    the palette and title match exactly.
    """
    weights = {
        "skus": 0.45,
        "categories": 0.20,
        "hex_codes": 0.15,
        "zones": 0.10,
        "budget_band": 0.05,
        "title_core": 0.05,
    }

    score = 0.0
    for key, weight in weights.items():
        left, right = a.get(key), b.get(key)
        if not left or not right:
            continue  # no evidence either way
        if key == "title_core":
            overlap = 1.0 if left == right else 0.0
        else:
            overlap = _jaccard(left, right)
        score += weight * overlap
    return round(score, 4)


def _jaccard(a: Any, b: Any) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_duplicate(a: Dict[str, Any], b: Dict[str, Any],
                 threshold: float = DUPLICATE_THRESHOLD) -> bool:
    return similarity(fingerprint(a), fingerprint(b)) >= threshold


def select_distinct(
    options: Sequence[Dict[str, Any]],
    limit: Optional[int] = None,
    threshold: float = DUPLICATE_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split options into the distinct ones and the rejected near-copies.

    Returns `(kept, rejected)` where each rejected entry carries its similarity
    to the option it duplicated, so the caller can explain the gap to the client
    rather than silently showing fewer options than requested.

    Order is preserved: the first occurrence of each distinct design wins, so
    this stays deterministic and does not shuffle what the pipeline produced.
    """
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    prints: List[Dict[str, Any]] = []

    for option in options or []:
        if not isinstance(option, dict):
            continue
        print_ = fingerprint(option)

        best_score = 0.0
        best_match: Optional[Dict[str, Any]] = None
        for existing in prints:
            score = similarity(print_, existing)
            if score > best_score:
                best_score, best_match = score, existing

        if best_match is not None and best_score >= threshold:
            rejected.append({
                "title": option.get("title", ""),
                "duplicate_of": best_match.get("title", ""),
                "similarity": best_score,
                "option": option,
            })
            continue

        prints.append({**print_, "title": option.get("title", "")})
        kept.append(option)

        if limit is not None and len(kept) >= limit:
            break

    return kept, rejected


def describe_rejection(rejection: Dict[str, Any]) -> str:
    """Client-facing explanation for why an option was not shown."""
    return (
        f"'{rejection.get('title', '')}' was dropped as a near-duplicate of "
        f"'{rejection.get('duplicate_of', '')}' "
        f"({rejection.get('similarity', 0):.0%} similar)."
    )