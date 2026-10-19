"""
Metadata inference for the knowledge base.

Why infer rather than hand-tag
------------------------------
The corpus is written as prose, and tagging 200+ passages by hand is exactly
the kind of bookkeeping that rots: a passage gets edited and its tags quietly
stop matching. Inferring the facets from the passage text keeps the metadata
in step with the content by construction, and it is deterministic, so the same
corpus always produces the same facets.

The facets are the ones the retriever can actually filter on. A `room` tag
that is wrong is worse than no tag at all, so the rules below are
conservative: a passage is only tagged when the evidence is strong, and an
untagged passage is still retrievable by any query. Filtering narrows the
candidate set; it never excludes knowledge outright.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# Words that place a passage in a room. Ordered longest-first so "home
# office" is not tagged as "office" before it can match the longer phrase.
ROOM_KEYWORDS: Dict[str, tuple] = {
    "kids_room": (
        "children's room", "kids room", "child's room", "nursery", "toddler",
        "baby", "children", "school bag", "cot",
    ),
    "home_office": (
        "home office", "study room", "work from home", "desk", "wfh",
    ),
    "living_room": (
        "living room", "sofa", "television", "tv unit", "coffee table",
        "family room", "lounge",
    ),
    "bedroom": ("bedroom", "bed", "wardrobe", "pillow", "duvet", "headboard"),
    "kitchen": ("kitchen", "cooking", "countertop", "sink", "tandoor", "pantry"),
    "dining": ("dining", "dining table", "banquet"),
    "bathroom": ("bathroom", "bath", "shower", "wash basin", "tiled"),
    "balcony": ("balcony", "terrace", "verandah"),
    "entrance": ("entrance", "doorway", "foyer", "entry", "vestibule"),
}

# Style vocabulary.
STYLE_KEYWORDS: Dict[str, tuple] = {
    "modern": ("modern", "contemporary", "minimalist furniture", "clean lines"),
    "minimalist": ("minimalist", "minimal", "decluttered", "less is more"),
    "scandinavian": ("scandinavian", "nordic", "scandi", "light wood", "hygge"),
    "bohemian": ("bohemian", "boho", "eclectic", "layered textiles", "jhumka"),
    "industrial": ("industrial", "exposed brick", "metal", "factory", "concrete"),
    "japanese": ("japanese", "zen", "wabi", "minimal japanese", "tatami"),
    "midcentury": ("mid-century", "midcentury", "retro", "teak", "walnut finish"),
    "luxury": ("luxury", "marble", "velvet", "imported", "premium", "brass"),
    "traditional_indian": (
        "traditional", "jaipur", "inlay", "carved", "brass", "block print",
        "handwoven", "khadi",
    ),
    "cozy": ("cozy", "cosy", "warm", "layered lighting", "soft textures"),
}

# Budget tiers, matched against explicit price anchors in the text.
BUDGET_KEYWORDS: Dict[str, tuple] = {
    "low": (
        "budget", "low-cost", "inexpensive", "under 20,000", "under 10,000",
        "affordable", "cheap", "thrift", "repurpos", "diy", "under 50,000",
    ),
    "medium": (
        "mid-range", "midrange", "50,000", "1,50,000", "1 lakh", "moderate",
        "value for money", "standard",
    ),
    "high": (
        "premium", "2 lakh", "5 lakh", "upper-mid", "high-end", "branded",
    ),
    "luxury": (
        "luxury", "ultra-premium", "imported", "bespoke", "10 lakh", "elite",
    ),
}

# Vastu-specific passages.
VASTU_KEYWORDS = (
    "vastu", "vastu shastra", "compass", "north-east", "northeast",
    "south-west", "southwest", "bed head", "prahastu", "remedy", "dosha",
    "direction of the", "solar",
)


def infer_metadata(content: str, category: str = "") -> Dict[str, object]:
    """
    Derive filterable facets from a passage.

    Returns a dict with `rooms`, `styles`, `budget_tiers`, `vastu` (bool),
    and `category`. Every list may be empty; an empty list means "no strong
    signal", not "does not apply", so the retriever treats it as unfiltered
    rather than excluding the passage.
    """
    text = (content or "").lower()

    rooms = [
        room for room, words in ROOM_KEYWORDS.items()
        if any(w in text for w in words)
    ]
    styles = [
        style for style, words in STYLE_KEYWORDS.items()
        if any(w in text for w in words)
    ]
    tiers = [
        tier for tier, words in BUDGET_KEYWORDS.items()
        if any(w in text for w in words)
    ]
    window_directions: List[str] = []
    direction_aliases = (
        ("northeast", ("northeast", "north east", "north-east")),
        ("northwest", ("northwest", "north west", "north-west")),
        ("southeast", ("southeast", "south east", "south-east")),
        ("southwest", ("southwest", "south west", "south-west")),
        ("north", ("north",)), ("south", ("south",)),
        ("east", ("east",)), ("west", ("west",)),
    )
    for canonical, aliases in direction_aliases:
        for alias in aliases:
            if re.search(
                rf"\b{re.escape(alias)}(?:[- ]facing)?\s+windows?\b", text
            ) or re.search(
                rf"\b{re.escape(alias)}\s+windows?\b", text
            ) or re.search(
                rf"\bwindows?\s+(?:face|faces|facing|on|to|in)\s+"
                rf"{re.escape(alias)}\b", text
            ):
                window_directions.append(canonical)
                break
    vastu = any(w in text for w in VASTU_KEYWORDS)

    return {
        "category": category,
        "rooms": rooms,
        "styles": styles,
        "budget_tiers": tiers,
        "window_directions": window_directions,
        "vastu": vastu,
    }


def expand_chunks(knowledge_base: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """
    Attach inferred metadata to every chunk in the base.

    Existing metadata is preserved and the inferred facets are merged into
    it, so a hand-tagged field always wins over an inferred one.
    """
    out: List[Dict[str, object]] = []
    for entry in knowledge_base:
        content = entry.get("content", "")
        category = entry.get("category", "")
        inferred = infer_metadata(content, category)
        merged: Dict[str, object] = {**inferred}
        for key, value in (entry.get("metadata") or {}).items():
            merged[key] = value
        merged["content"] = content
        merged["source"] = entry.get("source", category)
        merged["category"] = category
        out.append(merged)
    return out


def matches_filter(
    meta: Dict[str, object],
    room: Optional[str] = None,
    style: Optional[str] = None,
    budget_tier: Optional[str] = None,
    window_direction: Optional[str] = None,
    require_vastu: bool = False,
) -> bool:
    """
    Does a chunk satisfy the requested filters?

    Untagged chunks always pass. A passage with no room tag is general
    advice - "maintain 90 cm of circulation" applies everywhere - and
    excluding it because it lacks a tag would be exactly backwards.
    """
    if room:
        rooms = meta.get("rooms") or []
        if rooms and room not in rooms:
            return False
    if style:
        styles = meta.get("styles") or []
        if styles and style not in styles:
            return False
    if budget_tier:
        tiers = meta.get("budget_tiers") or []
        if tiers and budget_tier not in tiers:
            return False
    if window_direction:
        directions = meta.get("window_directions") or []
        requested = (
            window_direction.lower().replace("-", "").replace(" ", "")
        )
        if directions and requested not in directions:
            return False
    if require_vastu and not meta.get("vastu"):
        return False
    return True
