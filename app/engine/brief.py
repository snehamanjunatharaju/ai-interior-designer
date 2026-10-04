"""
Brief parsing: turn free text into hard, checkable constraints.

The central failure this fixes
-----------------------------
The original pipeline passed the client's brief to the LLM as a *prompt* and
hoped the model honoured it. A 2B model asked for "2 beds" cheerfully returns
one, reports "within budget", and nothing catches it. Language models are good
at writing; they are unreliable at *obeying*.

So the brief is parsed here, in code, into an explicit constraint set. Those
constraints are then checked against the produced design by
`app.engine.validator`, which has no access to the LLM and therefore no way to
rationalise its way past a failure.

Supported extractions
---------------------
- Quantities:     "2 beds", "two beds", "a pair of chairs", "3-bunk"
- Money:          "Rs 10,000", "10k", "1.5 lakh", "budget of 20000"
- Colour scheme:  "vibrant", "pastel", "monochrome", ...
- Room type:      inferred from keywords
- Directions:     "window faces east", "door on the north side"
- Dimensions:     "12 x 10 ft", "12ft x 10ft", "3.6m x 3m"
"""

from __future__ import annotations

import re
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.catalog.store import TIER_ESSENTIAL, CatalogItem

# --------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------

# Canonical item names mapped to the words that identify them in a brief.
ITEM_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "bed": ("bed", "cot", "bunk", "bedstead", "crib"),
    "mattress": ("mattress", "mat"),
    "desk": ("desk", "study table", "table", "workstation"),
    "chair": ("chair", "stool", "seating"),
    "wardrobe": ("wardrobe", "closet", "almirah", "almirah"),
    "bookshelf": ("bookshelf", "book shelf", "shelf", "bookcase", "shelving"),
    "sofa": ("sofa", "couch", "settee"),
    "armchair": ("armchair", "arm chair", "single chair", "lounge chair"),
    "beanbag": ("bean bag", "beanbag"),
    "tv_unit": ("tv unit", "tv stand", "television unit", "tv console", "tv"),
    "rug": ("rug", "carpet", "mat"),
    "curtain": ("curtain", "curtains", "drape", "blinds"),
    "mirror": ("mirror",),
    "storage": ("storage", "chest", "hamper", "stool"),
    "table": ("table", "side table", "coffee table"),
    "lighting": ("lamp", "light", "lighting"),
    "paint": ("paint", "wall colour", "wall color", "paint the wall"),
}

COLOR_SCHEMES: Dict[str, Tuple[str, ...]] = {
    "vibrant": ("vibrant", "bright", "bold", "colourful", "colorful", "playful", "lively"),
    "pastel": ("pastel", "soft pastel", "baby colours", "baby colors"),
    "neutral": ("neutral", "beige", "greige", "off white", "off-white"),
    "monochrome": ("monochrome", "black and white", "greyscale", "grayscale"),
    "earthy": ("earthy", "terracotta", "rust", "clay"),
    "cool": ("cool toned", "cool-toned", "blue", "bluer", "teal"),
    "warm": ("warm toned", "warm-toned", "warm"),
    "earthy_green": ("olive", "sage", "green"),
    "jewel": ("jewel tone", "jewel tones", "emerald", "sapphire"),
}

# Directions, normalised to full words.
DIRECTIONS = {
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
}

# Plurals and number words for quantity extraction.
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a": 1, "an": 1, "single": 1, "pair": 2, "couple": 2, "dozen": 12,
}

# Which items make sense in which room, so a "kids room" brief does not
# silently generate a kitchen. Values are the categories the catalog uses.
ROOM_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "living_room": ("sofa", "seating", "tv_unit", "rug", "table", "mirror", "lighting", "decor"),
    "bedroom": ("bed", "mattress", "wardrobe", "side_table", "table", "curtain", "rug", "lighting"),
    "kids_room": ("bed", "mattress", "desk", "chair", "wardrobe", "bookshelf", "storage", "curtain", "rug", "decor"),
    "home_office": ("desk", "chair", "bookshelf", "storage", "table", "lighting"),
    "dining": ("table", "chair", "mirror", "lighting", "decor"),
    "kitchen": ("kitchen",),
    "bathroom": ("mirror", "lighting", "decor"),
    "balcony": ("seating", "decor", "decor"),
    "entrance": ("decor", "mirror", "lighting"),
}

# Brief vocabulary -> catalog categories.
#
# A client says "sofa"; the catalog stores it under `seating`. Without this
# mapping the brief parser extracts "sofa", the catalog lookup finds nothing,
# and the design is rejected as missing an item that was in fact available.
# The mapping is the join between the two vocabularies, and it is the reason
# a living-room brief does not silently fail to find a sofa.
#
# The second element is a narrower set of *acceptable* sub-types, used where
# a category is too broad to count on its own. "seating" holds a sofa, an
# armchair, a bean bag and a stool; a stool is not a sofa, so a sofa request
# may only be satisfied by a sofa or a sectional.
BRIEF_TO_CATALOG_CATEGORY: Dict[str, Tuple[str, ...]] = {
    "sofa": ("seating",),
    "armchair": ("seating",),
    "beanbag": ("seating",),
    "chair": ("chair", "seating"),
    "bookshelf": ("storage",),
    "storage": ("storage",),
    "tv_unit": ("tv_unit",),
    "wardrobe": ("wardrobe",),
    "mattress": ("mattress",),
    "rug": ("soft",),
    "curtain": ("soft",),
    "table": ("table",),
    "lighting": ("lighting",),
    "mirror": ("decor",),
    "paint": ("paint",),
    "bed": ("bed",),
    "desk": ("desk",),
}

# Where the catalog category is too coarse, these say which specific product
# names may satisfy a brief term. Matched case-insensitively as substrings.
ACCEPTABLE_PRODUCTS: Dict[str, Tuple[str, ...]] = {
    # A stool or bean bag is seating, but it is not a sofa.
    "sofa": ("sofa", "sectional", "couch", "settee"),
    "armchair": ("armchair", "lounge chair", "single chair"),
    "beanbag": ("bean bag", "beanbag"),
    # A bookshelf may be a generic shelving unit. If the catalog has nothing
    # more specific, a plain shelf still serves the need.
    "bookshelf": ("bookshelf", "book shelf", "shelf", "shelving", "cube", "storage"),
    # A mirror is a decor-category product, but a planter is not a mirror.
    "mirror": ("mirror",),
}

CATEGORIES_WITH_QUANTITY = {"bed", "mattress", "chair", "sofa", "armchair",
                            "beanbag", "rug", "curtain", "mirror", "storage", "table"}

# Words that mean "no such item" - negation must not be read as a request.
NEGATIONS = ("no ", "not ", "without ", "avoid ", "don't ", "dont ", "skip ")

# A conjunction between two items means the next noun starts a new phrase, so
# a number must not be allowed to carry across it. Without this, "two beds and
# a desk" reads as "2 beds" *and* "2 desks".
ITEM_BOUNDARY_INNER = r"and|but|plus|also|with|of|for|or|the|a|an"
ITEM_BOUNDARY = r"(?:\b(?:and|but|plus|also|with|of|for|or)\b|[,;+]|&)"


@dataclass
class RoomDimensions:
    """Physical room dimensions. Defaults are conservative Indian apartment values."""

    length_ft: Optional[float] = None
    width_ft: Optional[float] = None
    length_cm: Optional[int] = None
    width_cm: Optional[int] = None
    ceiling_ft: Optional[float] = None
    ceiling_cm: Optional[int] = None

    @property
    def area_sqft(self) -> Optional[float]:
        if self.length_ft and self.width_ft:
            return round(self.length_ft * self.width_ft, 1)
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "length_ft": self.length_ft,
            "width_ft": self.width_ft,
            "length_cm": self.length_cm,
            "width_cm": self.width_cm,
            "ceiling_ft": self.ceiling_ft,
            "ceiling_cm": self.ceiling_cm,
            "area_sqft": self.area_sqft,
        }

    def is_known(self) -> bool:
        return self.length_ft is not None and self.width_ft is not None


def ft_to_cm(ft: float) -> int:
    return int(round(ft * 30.48))


# --------------------------------------------------------------------------
# Individual extractors
# --------------------------------------------------------------------------


def parse_money(text: str) -> Optional[int]:
    """
    Extract a budget in INR.

    Handles "10,000", "Rs 10,000", "10k", "1.5 lakh", "1 lakh", "15000/-".
    Returns the largest plausible figure if several appear, since a brief
    usually states the total budget rather than a component price.
    """
    if not text:
        return None

    candidates: List[int] = []

    # lakh / lac / cr variants
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|lakhs)\b", text, re.I):
        candidates.append(int(float(match.group(1)) * 100_000))
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:crore|cr)\b", text, re.I):
        candidates.append(int(float(match.group(1)) * 10_000_000))

    # Thousands shorthand: "10k", "10K". The trailing lookahead prevents a
    # non-money "k" (as in "10 km away") from being read as a budget.
    for match in re.finditer(r"(?<![\d.\w])(\d+(?:\.\d+)?)\s*k(?![a-z])", text, re.I):
        candidates.append(int(float(match.group(1)) * 1_000))

    # Plain rupee figures, optionally with Rs/INR/₹/commas
    for match in re.finditer(
        r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)|(?<![kK])\b([\d,]{4,})\b", text, re.I
    ):
        raw = match.group(1) or match.group(2)
        if not raw:
            continue
        try:
            value = int(float(raw.replace(",", "")))
        except ValueError:
            continue
        # Ignore implausible figures (years, dimensions already handled).
        if 500 <= value <= 100_000_000:
            candidates.append(value)

    return max(candidates) if candidates else None


def parse_quantity(text: str, item: str) -> Optional[int]:
    """
    Extract a requested count for one item type.

    Handles "2 beds", "two beds", "a pair of chairs", "3-bunk". The number
    must sit in the *same noun phrase* as the item, so an intervening
    conjunction ("two beds and a desk") does not leak the count onto the
    second item.

    Returns None when the item is not mentioned, so callers can distinguish
    "absent" from "explicitly zero".
    """
    number_alt = r"\d+|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    # Words allowed between a count and its noun: "a pair of", "2 single".
    # A conjunction is deliberately excluded, so a count cannot leak across
    # into the next noun phrase ("two beds and a desk" -> 2 beds, 1 desk).
    filler = r"(?:\s+(?:of|the|a|an|per|each))*\s*"
    # The noun may itself be compound: "study table", "tv unit".
    prefix = r"(?:[\w'-]+\s+){0,2}"

    # "a pair of chairs" / "a couple of stools" means two, not one. This has
    # to be handled before the generic path, which would read the "a" and
    # answer 1.
    for syn in ITEM_SYNONYMS.get(item, (item,)):
        for match in re.finditer(
            rf"\b(?:a|one)\s+(pair|couple|set)\s+of\s+(?:[\w'-]+\s+){{0,2}}{re.escape(syn)}s?\b",
            text, re.I,
        ):
            return 2
        for match in re.finditer(
            rf"\b(pair|couple)\s+of\s+(?:[\w'-]+\s+){{0,2}}{re.escape(syn)}s?\b",
            text, re.I,
        ):
            return 2

    for syn in ITEM_SYNONYMS.get(item, (item,)):
        # "<number> [filler] [compound noun] <synonym>"
        lead = rf"({number_alt}){filler}{prefix}{re.escape(syn)}s?\b"
        for match in re.finditer(lead, text, re.I):
            token = match.group(1).lower()
            if token.isdigit():
                return int(token)
            if token in NUMBER_WORDS:
                return NUMBER_WORDS[token]
        # "<item> x2"
        for match in re.finditer(
            rf"{re.escape(syn)}s?\s*(?:x|×|\*)\s*(\d+)\b", text, re.I
        ):
            return int(match.group(1))
    return None


def mentions_item(text: str, item: str) -> bool:
    """True if the item appears in the text outside a negated clause."""
    lower = text.lower()
    for syn in ITEM_SYNONYMS.get(item, (item,)):
        for match in re.finditer(rf"\b{re.escape(syn)}s?\b", lower):
            start = max(0, match.start() - 20)
            context = lower[start:match.start()]
            if not any(neg in context for neg in NEGATIONS):
                return True
    return False


def parse_color_scheme(text: str) -> Optional[str]:
    """Return the canonical colour-scheme keyword, or None."""
    lower = text.lower()
    for canonical, words in COLOR_SCHEMES.items():
        if any(w in lower for w in words):
            return canonical
    return None


def parse_direction(text: str) -> Optional[str]:
    """Return a cardinal/intercardinal direction mentioned, or None."""
    lower = text.lower()
    # Check intercardinal before cardinal so "northeast" is not read as "north".
    for direction in ("northeast", "northwest", "southeast", "southwest",
                      "north", "south", "east", "west"):
        if re.search(rf"\b{direction}\b", lower):
            return direction
    return None


def parse_directions(text: str) -> Dict[str, str]:
    """
    Extract per-opening directions: door and window.

    Handles "window faces east", "east facing window", "door on the north side",
    "north-facing door", and the terse "window: east".
    """
    lower = text.lower()
    found: Dict[str, str] = {}

    for opening in ("door", "window", "entrance"):
        # "<opening> ... <direction>" within a short window
        for match in re.finditer(
            rf"{opening}\w*\b[^.]{{0,40}}?\b(northeast|northwest|southeast|southwest|north|south|east|west)\b",
            lower,
        ):
            found.setdefault(opening, match.group(1))
        # "<direction> facing <opening>"
        for match in re.finditer(
            rf"\b(northeast|northwest|southeast|southwest|north|south|east|west)\b[^.]{{0,25}}?{opening}\w*\b",
            lower,
        ):
            found.setdefault(opening, match.group(1))
        # "window: east"
        for match in re.finditer(rf"{opening}\w*\s*[:\-]\s*\b(north|south|east|west)\b", lower):
            found.setdefault(opening, match.group(1))

    return found


def parse_dimensions(text: str) -> RoomDimensions:
    """
    Extract room dimensions.

    Handles "12 x 10 ft", "12ft x 10ft", "12' x 10'", "3.6m x 3m",
    "240cm x 200cm", "12ft x 10ft (2400mm x 3000mm)", "12ft 6in x 10ft",
    "ceiling 9ft", "height 9 ft". The longer number is treated as length.
    Feet, metres, centimetres, millimetres and inches are all accepted, because
    a brief copied from a builder or an architect's plan arrives in any of
    them and rejecting one would reject the whole request.
    """
    dims = RoomDimensions()
    if not text:
        return dims

    lower = text.lower()

    # Feet / ft / feet / inches-style pairs
    ft_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:'|’|ft|feet|foot)?\s*(?:x|×|\*)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:'|’|ft|feet|foot)",
        lower,
    )
    if ft_match:
        a, b = float(ft_match.group(1)), float(ft_match.group(2))
        dims.length_ft, dims.width_ft = max(a, b), min(a, b)
    else:
        # Metric pairs
        m_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:m|metre|meter)s?\s*(?:x|×|\*)\s*(\d+(?:\.\d+)?)\s*m\b",
            lower,
        )
        if m_match:
            a, b = float(m_match.group(1)), float(m_match.group(2))
            dims.length_ft = round(max(a, b) / 0.3048, 1)
            dims.width_ft = round(min(a, b) / 0.3048, 1)
        else:
            # Explicit sub-metre units: "240cm x 200cm", "1200mm x 1500mm",
            # and "12ft 6in". Checked before the unit-less form because 240 is
            # far too large to be feet.
            cm_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(cm|mm)\s*(?:x|×|\*)\s*(\d+(?:\.\d+)?)\s*(cm|mm)",
                lower,
            )
            if cm_match:
                # Honour each side's own unit rather than assuming they match.
                # Factor converts the side's unit into centimetres.
                to_cm = {"cm": 1.0, "mm": 0.1}
                a = float(cm_match.group(1)) * to_cm[cm_match.group(2)]
                b = float(cm_match.group(3)) * to_cm[cm_match.group(4)]
                dims.length_cm = int(max(a, b))
                dims.width_cm = int(min(a, b))
                dims.length_ft = round(dims.length_cm / 30.48, 1)
                dims.width_ft = round(dims.width_cm / 30.48, 1)
            else:
                inches_match = re.search(
                    r"(\d+(?:\.\d+)?)\s*(?:in\b|inches|inch|\")\s*(?:x|×|\*)\s*"
                    r"(\d+(?:\.\d+)?)\s*(?:in\b|inches|inch|\")",
                    lower,
                )
                if inches_match:
                    a, b = float(inches_match.group(1)), float(inches_match.group(2))
                    dims.length_cm = int(max(a, b) * 2.54)
                    dims.width_cm = int(min(a, b) * 2.54)
                    dims.length_ft = round(dims.length_cm / 30.48, 1)
                    dims.width_ft = round(dims.width_cm / 30.48, 1)
                else:
                    # A bare "10 x 12" with no unit. Ambiguous in principle,
                    # but Indian residential briefs mean feet almost
                    # universally, and a bare pair above 20 is implausible in
                    # either unit. Recorded as feet so the common case works
                    # rather than failing.
                    bare = re.search(
                        r"(?<![\d.])(\d{1,2}(?:\.\d)?)\s*(?:x|×|\*)\s*(\d{1,2}(?:\.\d)?)"
                        r"(?!\s*(?:cm|mm|m\b|in\b|ft|feet|foot|'))",
                        lower,
                    )
                    if bare:
                        a, b = float(bare.group(1)), float(bare.group(2))
                        if 3 <= max(a, b) <= 40:
                            dims.length_ft, dims.width_ft = max(a, b), min(a, b)

    # Ceiling / height
    ch_match = re.search(
        r"(?:ceiling|height)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*(?:'|’|ft|feet|foot)",
        lower,
    )
    if ch_match:
        dims.ceiling_ft = float(ch_match.group(1))
    else:
        ch_match_m = re.search(
            r"(?:ceiling|height)\s*(?:of|is|=|:)?\s*(\d+(?:\.\d+)?)\s*m\b", lower
        )
        if ch_match_m:
            dims.ceiling_ft = round(float(ch_match_m.group(1)) / 0.3048, 1)

    if dims.length_ft and dims.width_ft:
        # Only derive the centimetre values when they were not given directly.
        # A metric brief ("1200mm x 1500mm") is already exact, and round-tripping
        # it through feet loses precision - 150cm came back as 149.
        if dims.length_cm is None:
            dims.length_cm = ft_to_cm(dims.length_ft)
        if dims.width_cm is None:
            dims.width_cm = ft_to_cm(dims.width_ft)
    if dims.ceiling_ft:
        dims.ceiling_cm = ft_to_cm(dims.ceiling_ft)

    return dims


def catalog_categories_for(brief_term: Optional[str]) -> Tuple[str, ...]:
    """
    Catalog categories that satisfy a term used in the brief.

    The brief vocabulary and the catalog vocabulary are not the same: clients
    say "sofa", the catalog stores it as `seating`. Every lookup goes through
    this function so a request cannot be rejected for an item that is in
    fact in stock under a different name.
    """
    return BRIEF_TO_CATALOG_CATEGORY.get(brief_term or "", (brief_term or "",))


def find_products(term: Optional[str], limit: int = 0) -> List[CatalogItem]:
    """
    Resolve a brief term to real catalog products, cheapest viable first.

    Accepts None so callers can pass an optional brief term without guarding.
    `limit` caps the number returned; 0 means all matches.
    """
    from app.catalog import catalog

    if not term:
        return []
    catalog.initialize()
    results: List[CatalogItem] = []
    for category in catalog_categories_for(term):
        results.extend(catalog.by_category(category))
    if not results:
        return []
    # Cheapest first so an automatic pick never blows a tight budget.
    results.sort(key=lambda i: (i.tier != TIER_ESSENTIAL, i.price_inr))
    return results[:limit] if limit else results


def canonical_item_for(text: str) -> str:
    """
    Map free text ("the blue armchair", "study table") to a canonical term.

    Refinement requests name things casually, so the same substring-matching
    used for brief parsing is reused here. Returns a lowercase canonical name,
    or the trimmed input when nothing matches.
    """
    lower = (text or "").strip().lower()
    # Longest synonym first: "armchair" must beat "chair", and "bookshelf"
    # must beat "shelf", or the more specific item is never recognised.
    for item in ITEM_SYNONYMS:
        for syn in sorted(ITEM_SYNONYMS[item], key=len, reverse=True):
            if re.search(rf"\b{re.escape(syn)}s?\b", lower):
                return item
    return lower


def products_satisfying(term: str, limit: int = 0) -> List[CatalogItem]:
    """
    Products that genuinely satisfy a brief term, not merely share a category.

    `seating` contains sofas, armchairs, bean bags and stools. Treating the
    whole category as interchangeable would let a Rs 1,600 stool satisfy a
    request for a sofa, which is both a validation false negative and a way
    for an automatic insert to quietly under-deliver on the brief.
    """
    from app.catalog import catalog

    catalog.initialize()
    products = find_products(term)
    acceptable = ACCEPTABLE_PRODUCTS.get(term)
    if not acceptable:
        return products[:limit] if limit else products
    narrowed = [
        p for p in products
        if any(word in p.name.lower() for word in acceptable)
    ]
    # If nothing in the catalog matches the specific type, fall back to the
    # category rather than reporting the item as unavailable.
    chosen = narrowed or products
    return chosen[:limit] if limit else chosen


# --------------------------------------------------------------------------
# Brief -> constraints
# --------------------------------------------------------------------------


def infer_counts_from_occupants(bc: "BriefConstraints") -> None:
    """
    Infer bed counts from how many people the room is for.

    A brief routinely says "a room for two children" without ever using the
    word "bed". Without this step the design could return a single bed and
    still pass, because no explicit count was ever parsed. The rule is
    deliberately conservative: only sleeping rooms infer a bed per person,
    and an explicit count always wins.
    """
    sleeping_rooms = {"bedroom", "kids_room"}
    if bc.room_type not in sleeping_rooms:
        return

    people = 0

    for match in re.finditer(
        r"(\d+)\s+(?:young\s+)?(?:child|children|kid|kids|baby|babies"
        r"|toddler|toddlers|son|daughter|students?|girls?|boys?)",
        bc.raw_text, re.I,
    ):
        people = max(people, int(match.group(1)))

    for match in re.finditer(
        r"(" + "|".join(NUMBER_WORDS) + r")\s+(?:persons?|people|occupants?"
        r"|kids?|children|students?|teens?|adults?|guests?)",
        bc.raw_text, re.I,
    ):
        people = max(people, NUMBER_WORDS.get(match.group(1).lower(), 1))

    for match in re.finditer(
        r"(?:for|sleeps?|sleeping|accommodates?)\s+(\d+)\b",
        bc.raw_text, re.I,
    ):
        people = max(people, int(match.group(1)))

    if people:
        bc.occupant_count = people
        # An explicit "2 beds" always wins over an inferred count.
        bc.required_items.setdefault("bed", people)


@dataclass
class BriefConstraints:
    """
    Hard requirements extracted from the client brief.

    This is the contract the validator checks against. It is derived purely
    from text by deterministic code, so it is exactly as reliable as the
    regexes - which is why each extractor is written defensively and the
    raw matched text is retained for debugging.
    """

    room_type: str = ""
    style: str = ""
    color_scheme: Optional[str] = None
    budget_inr: Optional[int] = None
    dimensions: RoomDimensions = field(default_factory=RoomDimensions)
    vastu_compliance: str = "soft"

    # item -> requested count, only for items explicitly mentioned
    required_items: Dict[str, int] = field(default_factory=dict)
    # items mentioned with no count given
    optional_items: List[str] = field(default_factory=list)
    # items the brief explicitly says to avoid
    excluded_items: List[str] = field(default_factory=list)

    door_direction: Optional[str] = None
    window_direction: Optional[str] = None
    # Number of people the room must serve, inferred from the brief.
    occupant_count: Optional[int] = None

    occupants: List[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "room_type": self.room_type,
            "style": self.style,
            "color_scheme": self.color_scheme,
            "budget_inr": self.budget_inr,
            "dimensions": self.dimensions.to_dict(),
            "vastu_compliance": self.vastu_compliance,
            "required_items": dict(self.required_items),
            "optional_items": list(self.optional_items),
            "excluded_items": list(self.excluded_items),
            "door_direction": self.door_direction,
            "window_direction": self.window_direction,
            "occupant_count": self.occupant_count,
            "occupants": list(self.occupants),
        }

    def is_checkable(self) -> bool:
        """True if there is enough parsed detail to validate anything."""
        return bool(self.required_items or self.budget_inr or self.color_scheme)


def parse_brief(
    client_preferences: str,
    furniture: Optional[List[str]] = None,
    specific_requirements: Optional[List[str]] = None,
    occupants: Optional[List[str]] = None,
    room_type: str = "",
    style: str = "",
    color_scheme: str = "",
    budget_inr: Optional[int] = None,
    vastu_compliance: str = "soft",
    room_dimensions: str = "",
) -> BriefConstraints:
    """
    Parse a full brief into checkable constraints.

    Sources are merged in priority order: the free-text brief is the richest
    signal, then the structured furniture/requirement lists, then the
    explicit form fields (which the user selected deliberately and so
    override anything guessed from prose).
    """
    furniture = furniture or []
    specific_requirements = specific_requirements or []
    occupants = occupants or []

    # Concatenate everything the client said. Specific requirements are
    # included because "2 beds" is often written there, not in the prose.
    text = " ".join(
        part for part in [client_preferences or ""] + furniture
        + specific_requirements + occupants + [room_dimensions or ""]
        if part
    )

    bc = BriefConstraints(
        room_type=room_type or "",
        style=style or "",
        vastu_compliance=vastu_compliance,
        occupants=list(occupants),
        raw_text=text,
    )

    # -- money: explicit field wins, then prose --------------------------
    bc.budget_inr = budget_inr if budget_inr else parse_money(text)

    # -- colour: explicit select wins, else infer ------------------------
    if color_scheme and color_scheme.lower() != "neutral":
        bc.color_scheme = color_scheme.lower()
    else:
        inferred = parse_color_scheme(text)
        bc.color_scheme = inferred or (color_scheme.lower() if color_scheme else None)

    # -- dimensions -------------------------------------------------------
    bc.dimensions = parse_dimensions(text)

    # -- directions -------------------------------------------------------
    directions = parse_directions(text)
    bc.door_direction = directions.get("door") or directions.get("entrance")
    bc.window_direction = directions.get("window")

    # -- item quantities --------------------------------------------------
    # Iterate over the canonical item vocabulary so "2 beds" -> {"bed": 2}.
    for item in ITEM_SYNONYMS:
        qty = parse_quantity(text, item)
        if qty is not None and qty > 0:
            bc.required_items[item] = qty
        elif mentions_item(text, item):
            bc.optional_items.append(item)

    # Explicitly excluded items ("no mirror", "without a wardrobe") are
    # recorded separately so the validator can treat their absence as correct
    # and their presence as a violation.
    lower = text.lower()
    for item in ITEM_SYNONYMS:
        for syn in ITEM_SYNONYMS[item]:
            for match in re.finditer(rf"\b{re.escape(syn)}s?\b", lower):
                start = max(0, match.start() - 24)
                before = lower[start:match.start()]
                if any(before.rstrip().endswith(neg.strip()) for neg in NEGATIONS):
                    bc.excluded_items.append(item)
                    break

    # Furniture listed explicitly without a count still count as required.
    for f in furniture:
        for item in ITEM_SYNONYMS:
            if f.strip().lower() in [s.lower() for s in ITEM_SYNONYMS[item]]:
                bc.required_items.setdefault(item, 1)

    # De-duplicate optionals and drop anything already required or excluded.
    seen = set()
    bc.optional_items = [
        i for i in bc.optional_items
        if not (i in seen or i in bc.required_items or i in bc.excluded_items or seen.add(i))
    ]
    bc.excluded_items = sorted(set(bc.excluded_items) - set(bc.required_items))

    # Last: infer a bed per occupant where the brief implies it.
    infer_counts_from_occupants(bc)

    return bc
