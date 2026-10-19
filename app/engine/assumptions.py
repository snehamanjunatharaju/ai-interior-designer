"""
Claims the system is not entitled to make.

Three requirements in the brief share one root cause. The pipeline asserted
things it had not established:

  * "Strict Vastu" produced a compliance score and a note about natural light
    with no door or window bearing, so the score described a room whose
    orientation was invented;
  * a requested item could be absent from the shopping list while the prose
    described it in detail;
  * the layout description mentioned a mirror or a pendant light that was never
    bought, so the client furnished to a description they could not order.

All three are the same failure: a confident sentence that no code had checked.
This module makes the unverified claims explicit. `assess_assumptions` returns
the list, the pipeline surfaces it in the response, and the validator fails any
design that asserts more than it can support.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Compass points the system understands.
DIRECTIONS = {
    "north", "northeast", "north-east", "ne",
    "east", "southeast", "south-east", "se",
    "south", "southwest", "south-west", "sw",
    "west", "northwest", "north-west", "nw",
}

# Phrases that assert a bearing-dependent outcome. Each requires a real
# measurement to justify, which is why they are checked rather than trusted.
_BEARING_CLAIMS = re.compile(
    r"\b(natural light|daylight|sunlight|sun light|plenty of light|"
    r"well[- ]lit|well lit|good lighting|bright(ly)? (lit|room)|"
    r"vastu[- ]compliant|vastu[- ]friendly|vastu compliance|fully vastu|"
    r"strictly vastu|vastu compliant|respects? vastu|as per vastu|"
    r"according to vastu|(?:to )?comply with vastu|complies with vastu|"
    r"in accordance with vastu|prosperous direction)\b",
    re.I,
)

# Items a design may mention in prose. Each must appear in the shopping list if
# the prose recommends buying it.
_MENTIONED_ITEMS: Dict[str, Tuple[str, ...]] = {
    "mirror": ("mirror",),
    "lighting": ("lamp", "light", "pendant", "chandelier", "spotlight"),
    "rug": ("rug", "carpet"),
    "curtain": ("curtain", "drape", "blind"),
    "artwork": ("artwork", "wall art", "painting", "frame"),
    "plant": ("plant", "planter"),
}


def normalise_direction(value: Optional[str]) -> Optional[str]:
    """Canonicalise a compass direction, or return None if it is not one."""
    if not value:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in DIRECTIONS:
        # Try the long form for abbreviations like "n".
        long_forms = {
            "n": "north", "e": "east", "s": "south", "w": "west",
            "ne": "northeast", "se": "southeast",
            "sw": "southwest", "nw": "northwest",
        }
        text = long_forms.get(text, text)
        if text not in DIRECTIONS:
            return None
    return text


def has_bearings(door: Optional[str], window: Optional[str]) -> bool:
    return bool(normalise_direction(door) and normalise_direction(window))


def bearing_claims(text: str) -> List[str]:
    """Which bearing-dependent claims does this prose make?"""
    return [m.group(0) for m in _BEARING_CLAIMS.finditer(text or "")]


def prose_of(design: Dict[str, Any]) -> str:
    """Every free-text field of a design, joined for claim scanning."""
    parts: List[str] = []
    for key in (
        "concept", "layout_description", "lighting_plan",
        "storage_strategy", "vastu_notes",
    ):
        value = design.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(v) for v in value)
    for key in ("zones", "decor_suggestions", "pros", "cons"):
        value = design.get(key) or []
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, str):
                    parts.append(entry)
                elif isinstance(entry, dict):
                    parts.extend(
                        str(v) for v in entry.values() if isinstance(v, str)
                    )
    return "\n".join(parts)


def mentioned_items(design: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Items the prose talks about, grouped by kind.

    Used to check the text against the shopping list: prose that recommends a
    mirror the design never buys is a client-facing inconsistency.
    """
    text = prose_of(design).lower()
    found: Dict[str, List[str]] = {}
    for kind, words in _MENTIONED_ITEMS.items():
        hits = [w for w in words if re.search(rf"\b{re.escape(w)}", text)]
        if hits:
            found[kind] = sorted(set(hits))
    return found


def bought_categories(design: Dict[str, Any]) -> List[str]:
    """Categories actually present in the shopping list, lowercased."""
    out = set()
    for entry in design.get("furniture") or []:
        if isinstance(entry, dict):
            category = str(entry.get("category", "")).strip().lower()
            if category:
                out.add(category)
            # Category is not always set; fall back to matching the name.
            name = str(entry.get("name", "")).strip().lower()
            if name:
                out.add(name)
                for word in name.split():
                    if len(word) > 3:
                        out.add(word)
    return sorted(out)


# Which category satisfies a mention. A mirror is decor; a lamp is lighting.
_MENTION_SATISFIED_BY = {
    "mirror": "decor",
    "lighting": "lighting",
    "rug": "soft",
    "curtain": "soft",
    "artwork": "decor",
    "plant": "decor",
}


def unsupported_claims(design: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Claims in the prose that the rest of the design does not support.

    Two independent checks:

      * a bearing-dependent claim ("plenty of natural light") with no door or
        window direction on the design at all;
      * a concrete item ("wall mirror") named in prose but absent from the
        shopping list.

    Both return structured findings rather than a boolean so the message can
    name what is wrong, which is the only useful form for the client.
    """
    problems: List[Dict[str, str]] = []
    text = prose_of(design)

    door = normalise_direction(design.get("door_direction"))
    window = normalise_direction(design.get("window_direction"))

    claims = bearing_claims(text)
    if claims and not (door and window):
        missing = []
        if not door:
            missing.append("door direction")
        if not window:
            missing.append("window direction")
        problems.append({
            "code": "unverified_bearing_claim",
            "severity": "warning",
            "message": (
                f"The design claims {', '.join(sorted(set(claims)))} but "
                f"{' and '.join(missing)} "
                f"{'was' if len(missing) == 1 else 'were'} not supplied. "
                f"Without a bearing the claim cannot be verified."
            ),
            "remedy": (
                "Supply the door and window directions, or remove the claim. "
                "The layout below uses an assumed orientation."
            ),
        })

    # --- Vastu compliance claims vs the rule check ----------------------
    # The layout said "respects Vastu" while the code check reported the
    # cooking rule failed. Text must not outrank geometry: if the assessment
    # was run and found failures, a compliance claim is a false statement.
    assessment = design.get("_vastu") or {}
    results = assessment.get("results") or []
    vastu_claims = [
        c for c in claims
        if re.search(r"vastu", c, re.I)
    ]
    failed_rules = [r for r in results if not r.get("passed")]
    if vastu_claims and failed_rules:
        problems.append({
            "code": "contradicted_vastu_claim",
            "severity": "error",
            "message": (
                f"The design says {', '.join(sorted(set(vastu_claims)))}, but the "
                f"rule check reports {len(failed_rules)} failed rule(s): "
                + ", ".join(
                    str(r.get("title") or r.get("code")) for r in failed_rules[:3]
                )
                + "."
            ),
            "remedy": (
                "Remove the compliance claim. Vastu notes are generated from the "
                "rule-check results; describe the rules that passed instead."
            ),
        })

    bought = bought_categories(design)
    for kind, words in mentioned_items(design).items():
        required = _MENTION_SATISFIED_BY.get(kind)
        if not required:
            continue
        if any(required == c for c in bought):
            continue
        problems.append({
            "code": "mentioned_not_bought",
            "severity": "warning",
            "message": (
                f"The design text mentions a {kind} ({', '.join(words)}) but "
                f"no {required} item appears in the shopping list."
            ),
            "remedy": (
                f"Add a {required} line item, or remove the mention. A client "
                f"furnishing to the description cannot buy what it describes."
            ),
        })

    return problems


def assess_assumptions(
    design: Dict[str, Any],
    brief: Any = None,
    strict_vastu: bool = False,
) -> Dict[str, Any]:
    """
    Everything the design assumed, and whether it can be trusted.

    Returned on every response so the client can see the difference between
    "this was measured" and "this was inferred".
    """
    door = normalise_direction(
        getattr(brief, "door_direction", None) or design.get("door_direction")
    )
    window = normalise_direction(
        getattr(brief, "window_direction", None) or design.get("window_direction")
    )

    assumptions: List[Dict[str, str]] = []
    if door is None:
        assumptions.append({
            "field": "door_direction",
            "assumed": "north",
            "reason": "Not supplied and not stated in the brief.",
        })
    if window is None:
        assumptions.append({
            "field": "window_direction",
            "assumed": "east",
            "reason": "Not supplied and not stated in the brief.",
        })

    dimensions = design.get("room_dimensions") or design.get("room") or {}
    if not dimensions:
        assumptions.append({
            "field": "room_dimensions",
            "assumed": "none",
            "reason": "No room size was supplied, so no floor plan can be trusted.",
        })

    audience = getattr(brief, "audience", "any")
    if audience == "any":
        assumptions.append({
            "field": "occupants",
            "assumed": "unspecified",
            "reason": (
                "Occupants were not stated, so the design was not filtered by "
                "who the room is for."
            ),
        })

    bearings_known = bool(door and window)
    problems = unsupported_claims(design)

    return {
        "assumptions": assumptions,
        "has_bearings": bearings_known,
        "door_direction": door,
        "window_direction": window,
        # Strict Vastu is a promise of compliance. Without bearings there is
        # nothing to be compliant with, so the claim must be withheld rather
        # than scored against an invented orientation.
        "vastu_verifiable": bearings_known,
        "vastu_withheld": bool(strict_vastu and not bearings_known),
        "unsupported_claims": problems,
        "clean": not assumptions and not problems,
    }


def missing_requirements(
    design: Dict[str, Any],
    requested: Iterable[str],
) -> List[Dict[str, str]]:
    """
    Requested items that are absent from the shopping list.

    This is a code check over the catalog, not a reading of the prose. The
    brief's own list is the authority, and the shopping list is compared against
    it directly, so a design cannot satisfy the request by describing the item
    beautifully while not buying it.
    """
    from app.engine.brief import products_satisfying

    bought = bought_categories(design)
    missing: List[Dict[str, str]] = []

    for term in requested or []:
        term = str(term).strip().lower()
        if not term:
            continue
        options = products_satisfying(term)
        if not options:
            # Nothing in the catalog serves this term; that is a catalog gap,
            # not a design failure, so it is reported for information.
            missing.append({
                "item": term,
                "reason": "no catalog product serves this request",
                "fatal": "false",
            })
            continue

        # Satisfied when any bought item is one of the products that would
        # answer this request, by SKU, name or category.
        bought_skus = {
            str(e.get("sku", "")).strip()
            for e in (design.get("furniture") or [])
            if isinstance(e, dict)
        }
        satisfied = False
        for option in options:
            if option.sku in bought_skus:
                satisfied = True
                break
            if option.name.lower() in bought:
                satisfied = True
                break
            if option.category.lower() in bought:
                satisfied = True
                break
        if not satisfied:
            missing.append({
                "item": term,
                "reason": "requested but absent from the shopping list",
                "fatal": "true",
            })

    return missing