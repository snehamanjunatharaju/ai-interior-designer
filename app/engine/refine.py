"""
Targeted refinement: change one part of a design, leave the rest alone.

The problem this solves
-----------------------
The original system could only regenerate a whole design. Asking for "make the
desk bigger" re-rolled everything: the palette moved, the bed changed, the
cost was restated. The client had to re-approve decisions they had already
signed off, and the usual reaction was to stop trusting the output.

So refinement is classified before anything is regenerated. The request is
parsed into an intent and a set of targets, and only the targeted section is
rebuilt. Everything else is carried across byte-for-byte.

This module owns the classification and the merge. The LLM, when available,
writes the replacement prose for the targeted part only.
"""

from __future__ import annotations

import re
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.observability import get_logger, metrics

logger = get_logger(__name__)

# What a refinement request can target.
TARGET_COLOR = "color"
TARGET_ITEM = "item"
TARGET_FURNITURE = "furniture"
TARGET_LAYOUT = "layout"
TARGET_BUDGET = "budget"
TARGET_LIGHTING = "lighting"
TARGET_UNKNOWN = "unknown"

# Colour words the client might type, mapped to canonical scheme names and
# concrete swatches. The swatches matter: a request for "bluer" has to resolve
# to actual hex values or the result is a vague adjective.
COLOR_WORDS: Dict[str, Dict[str, Any]] = {
    "blue": {"scheme": "cool", "hex": ["#1E3A5F", "#4A9FD8", "#2E6F9E"], "names": ["navy", "sky blue", "teal blue"]},
    "bluer": {"scheme": "cool", "hex": ["#1E3A5F", "#4A9FD8", "#2E6F9E"], "names": ["navy", "sky blue", "teal blue"]},
    "green": {"scheme": "earthy_green", "hex": ["#7A8B4F", "#4F7942", "#A3B18A"], "names": ["olive", "forest", "sage"]},
    "red": {"scheme": "warm", "hex": ["#A63D2F", "#C1440E", "#7B2D26"], "names": ["brick red", "terracotta red", "maroon"]},
    "pink": {"scheme": "pastel", "hex": ["#F7C6D0", "#E88D9B", "#FBD9E1"], "names": ["rose", "blush", "soft pink"]},
    "purple": {"scheme": "cool", "hex": ["#5B3A70", "#7E5A9B", "#3D2A4D"], "names": ["plum", "violet", "aubergine"]},
    "yellow": {"scheme": "warm", "hex": ["#F2A93B", "#E8C547", "#D4A017"], "names": ["marigold", "mustard", "gold"]},
    "orange": {"scheme": "warm", "hex": ["#E07A3F", "#C1662F", "#F0A868"], "names": ["terracotta", "burnt orange", "apricot"]},
    "grey": {"scheme": "monochrome", "hex": ["#5A5A5A", "#8A8A8A", "#C4C4C4"], "names": ["charcoal", "grey", "silver"]},
    "gray": {"scheme": "monochrome", "hex": ["#5A5A5A", "#8A8A8A", "#C4C4C4"], "names": ["charcoal", "grey", "silver"]},
    "white": {"scheme": "neutral", "hex": ["#FFFFFF", "#F2EFE9", "#E8E4DC"], "names": ["white", "off white", "bone"]},
    "black": {"scheme": "monochrome", "hex": ["#1A1A1A", "#333333", "#4D4D4D"], "names": ["black", "charcoal", "graphite"]},
    "beige": {"scheme": "neutral", "hex": ["#D9CFC0", "#C8BCA8", "#EDE6DA"], "names": ["beige", "sand", "cream"]},
    "earthy": {"scheme": "earthy", "hex": ["#C0704A", "#7A8B4F", "#A63D2F"], "names": ["terracotta", "olive", "rust"]},
    "vibrant": {"scheme": "vibrant", "hex": ["#FF6B5B", "#17A2A2", "#F2A93B"], "names": ["coral", "teal", "marigold"]},
    "pastel": {"scheme": "pastel", "hex": ["#F7C6D0", "#BBD7EE", "#BFEAD9"], "names": ["rose", "powder blue", "mint"]},
    "neutral": {"scheme": "neutral", "hex": ["#F2EFE9", "#CFC7BA", "#9CAF88"], "names": ["warm white", "greige", "sage"]},
}

# Item words a refinement might name.
ITEM_WORDS = {
    "desk", "table", "bed", "cot", "chair", "stool", "sofa", "armchair",
    "wardrobe", "shelf", "bookshelf", "storage", "rug", "carpet", "curtain",
    "curtains", "mirror", "lamp", "light", "tv", "bedside", "beanbag", "bean bag",
}

# Longer words must be matched before their substrings. Without this,
# "armchair" matches "chair" first, so "swap the armchair for a bean bag"
# would target the chair and leave the armchair in the room.
ITEM_WORD_ORDER = tuple(sorted(ITEM_WORDS, key=len, reverse=True))

# Verbs that signal swapping one item for another.
SWAP_RE = re.compile(
    r"\b(?:swap|replace|change|instead of|exchange|substitute)\b", re.I
)
MOVE_RE = re.compile(r"\b(?:move|shift|reposition|put|relocate)\b", re.I)
ADD_RE = re.compile(r"\b(?:add|include|bring in|get|buy)\b", re.I)
REMOVE_RE = re.compile(r"\b(?:remove|delete|drop|take out|get rid of)\b", re.I)
SIZE_RE = re.compile(r"\b(?:bigger|larger|smaller|wider|narrower|taller|smaller)\b", re.I)
COLOR_RE = re.compile(
    r"\b(?:bluer|greener|redder|pinker|make it|more|less|colour|color|paint)\b",
    re.I,
)


@dataclass
class RefinementIntent:
    """What the client asked to change, parsed from plain English."""

    target: str
    original_text: str
    # e.g. {"color": "blue"} or {"item": "armchair"}
    slots: Dict[str, Any] = field(default_factory=dict)
    item_names: List[str] = field(default_factory=list)
    color_key: Optional[str] = None
    confidence: float = 0.0
    # The sentence the LLM should rewrite, when the target is prose.
    prose_field: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "original_text": self.original_text,
            "slots": self.slots,
            "items": self.item_names,
            "color": self.color_key,
            "confidence": self.confidence,
            "prose_field": self.prose_field,
        }


def classify_refinement(text: str) -> RefinementIntent:
    """
    Work out what part of the design the client is asking to change.

    Deliberately conservative and rule-based. Misclassifying a request is
    costly: a "make the desk bigger" handled as a full redesign rewrites the
    whole room, which is precisely the behaviour this module exists to stop.
    When nothing matches, the target is `unknown` and the caller can ask
    rather than guess.
    """
    raw = (text or "").strip()
    lower = raw.lower()

    color_key = next((c for c in COLOR_WORDS if c in lower), None)
    # Match longest first, then drop any word whose match lies inside an
    # already-claimed span. "armchair" contains "chair", so without this the
    # request "swap the armchair for a bean bag" yields three items instead of
    # two, and the chair gets removed as well.
    items: List[str] = []
    claimed: List[tuple] = []
    for word in ITEM_WORD_ORDER:
        for m in re.finditer(rf"\b{re.escape(word)}\b", lower):
            span = (m.start(), m.end())
            if any(span[0] >= c[0] and span[1] <= c[1] for c in claimed):
                continue
            claimed.append(span)
            items.append(word)
            break

    # Colour-only request: strongest signal, check first.
    if color_key and not (SWAP_RE.search(lower) or REMOVE_RE.search(lower)):
        if items and SIZE_RE.search(lower):
            target = TARGET_ITEM
        else:
            target = TARGET_COLOR
            return RefinementIntent(
                target=target, original_text=raw, color_key=color_key,
                slots={"color": color_key}, confidence=0.9,
            )

    if SWAP_RE.search(lower) and items:
        # The source is the noun closest to the verb, i.e. the first item word
        # in the text *after* the verb. Sorting globally by length picked
        # "bean bag" (the replacement) and swapped out the wrong piece.
        after_verb = SWAP_RE.split(lower, maxsplit=1)[-1]
        source = None
        best_pos = None
        for word in ITEM_WORD_ORDER:
            m = re.search(rf"\b{re.escape(word)}\b", after_verb)
            if m and (best_pos is None or m.start() < best_pos):
                source, best_pos = word, m.start()
        return RefinementIntent(
            target=TARGET_ITEM, original_text=raw, item_names=items,
            slots={"action": "swap", "from": source or items[0]}, confidence=0.85,
        )

    if REMOVE_RE.search(lower) and items:
        return RefinementIntent(
            target=TARGET_ITEM, original_text=raw, item_names=items,
            slots={"action": "remove"}, confidence=0.85,
        )

    # "add a reading nook" names an outcome, not a catalog item. Checked
    # before the generic add path, which would otherwise look for an item
    # called "nook" and find none.
    if ADD_RE.search(lower) and any(
        w in lower for w in ("nook", "corner", "space", "area", "zone", "cozy")
    ):
        outcome = next(
            (w for w in ("reading nook", "corner", "space", "zone", "cozy corner")
             if w in lower),
            "area",
        )
        return RefinementIntent(
            target=TARGET_ITEM, original_text=raw,
            item_names=["chair"],
            slots={"action": "add", "outcome": outcome,
                   "hint": "a reading nook is a chair plus a side table and a lamp"},
            confidence=0.6,
        )

    if ADD_RE.search(lower) and items:
        return RefinementIntent(
            target=TARGET_ITEM, original_text=raw, item_names=items,
            slots={"action": "add"}, confidence=0.8,
        )

    if MOVE_RE.search(lower) and items:
        return RefinementIntent(
            target=TARGET_LAYOUT, original_text=raw, item_names=items,
            slots={"action": "move"}, confidence=0.75,
        )

    if SIZE_RE.search(lower) and items:
        bigger = any(w in lower for w in ("bigger", "larger", "wider", "taller"))
        return RefinementIntent(
            target=TARGET_ITEM, original_text=raw, item_names=items,
            slots={"action": "resize", "bigger": bigger}, confidence=0.8,
        )

    if "light" in lower or "lamp" in lower or "bright" in lower:
        return RefinementIntent(
            target=TARGET_LIGHTING, original_text=raw,
            slots={"action": "relight"}, confidence=0.6,
            prose_field="lighting_plan",
        )

    if "budget" in lower or "cheaper" in lower or "expensive" in lower:
        return RefinementIntent(
            target=TARGET_BUDGET, original_text=raw,
            slots={"action": "reprice"}, confidence=0.6,
        )

    if "layout" in lower or "arrangement" in lower or "furnish" in lower:
        return RefinementIntent(
            target=TARGET_LAYOUT, original_text=raw,
            slots={"action": "rearrange"}, confidence=0.6,
        )

    return RefinementIntent(target=TARGET_UNKNOWN, original_text=raw, confidence=0.0)


# --------------------------------------------------------------------------
# Applying the refinement
# --------------------------------------------------------------------------


def apply_refinement(
    design: Dict[str, Any],
    intent: RefinementIntent,
    catalog_lookup=None,
) -> Dict[str, Any]:
    """
    Apply a refinement to a design, mutating only the targeted part.

    Returns the same design dict for convenience, but every non-targeted field
    is left exactly as it was - that is the contract. Callers can diff the two
    to show the client precisely what moved.
    """
    if catalog_lookup is None:
        from app.engine.brief import products_satisfying as _lookup
        catalog_lookup = _lookup

    changed: List[str] = []

    if intent.target == TARGET_COLOR and intent.color_key:
        spec = COLOR_WORDS[intent.color_key]
        before = (design.get("color_palette") or {}).get("hex_codes")
        design["color_palette"] = {
            "primary": spec["names"][0],
            "secondary": spec["names"][1] if len(spec["names"]) > 1 else spec["names"][0],
            "accent": spec["names"][2] if len(spec["names"]) > 2 else spec["names"][0],
            "neutrals": ["white", "soft grey"],
            "hex_codes": list(spec["hex"]),
        }
        if before != spec["hex"]:
            changed.append("color_palette")
        # The prose often names the old colours, so refresh the concept line
        # only - not the whole design.
        concept = design.get("concept", "")
        if concept:
            design["concept"] = re.sub(
                r"\b(?:coral|teal|marigold|beige|terracotta|olive|sage)\b",
                spec["names"][0], concept, count=1, flags=re.I,
            )
            changed.append("concept")

    elif intent.target == TARGET_ITEM and intent.slots.get("action") in {"swap", "remove"}:
        from app.engine.brief import canonical_item_for as _canon
        furniture = design.get("furniture", [])
        if isinstance(furniture, list):
            # For a swap, only the source item is removed - never the noun the
            # client is swapping *to*, which is also mentioned in the sentence.
            sources = [intent.slots["from"]] if intent.slots.get("action") == "swap" \
                and intent.slots.get("from") else intent.item_names
            found_any = False
            for word in sources:
                canonical = _canon(word)
                before_len = len(furniture)
                if any(
                    isinstance(f, dict) and _canon(str(f.get("name", ""))) == canonical
                    for f in furniture
                ):
                    found_any = True
                # Compare on the canonical term in both directions: the
                # client's word ("armchair") and the catalog name
                # ("Armchair - Single") canonicalise identically, so an
                # exact equality test silently matched nothing and the item
                # was never removed.
                furniture = [
                    f for f in furniture
                    if not (isinstance(f, dict) and _canon(str(f.get("name", ""))) == canonical)
                ]
                if len(furniture) != before_len:
                    changed.append(f"removed {canonical}")

            if not found_any:
                # "Remove the desk" in a room with no desk is a no-op with an
                # explanation, not a silent success.
                _refuse(design, sources[0] if sources else "item", catalog_lookup)

            if intent.slots.get("action") == "swap" and found_any:
                # The replacement is an item word that is not the source. Note
                # that "bean bag" canonicalises to "beanbag" while "armchair"
                # canonicalises to "armchair", so the source is excluded by
                # word identity, not by canonical form.
                source_word = intent.slots.get("from")
                others = [w for w in intent.item_names if w != source_word]
                wanted = others[-1] if others else intent.item_names[-1]
                from app.engine.brief import canonical_item_for as _canon
                if _canon(wanted) == _canon(source_word or ""):
                    wanted = others[0] if others else wanted
                options = catalog_lookup(_canon(wanted), limit=1)
                if options:
                    product = options[0]
                    furniture.append({
                        "name": product.name,
                        "category": product.category,
                        "sku": product.sku,
                        "quantity": 1,
                        "brand_suggestions": [product.brand],
                    })
                    changed.append(f"swapped in {product.name}")
            design["furniture"] = furniture

    elif intent.target == TARGET_ITEM and intent.slots.get("action") == "add":
        from app.engine.brief import canonical_item_for as _canon
        wanted = intent.item_names[-1]
        options = catalog_lookup(_canon(wanted), limit=1)
        if options:
            product = options[0]
            furniture = design.setdefault("furniture", [])
            if isinstance(furniture, list):
                furniture.append({
                    "name": product.name,
                    "category": product.category,
                    "sku": product.sku,
                    "quantity": 1,
                    "brand_suggestions": [product.brand],
                })
                changed.append(f"added {product.name}")
        else:
            _refuse(design, wanted, catalog_lookup)

    elif intent.target == TARGET_ITEM and intent.slots.get("action") == "resize":
        # Resizing is a catalog concern: pick a larger or smaller variant of
        # the same product type so the price stays real.
        bigger = bool(intent.slots.get("bigger"))
        current_word = intent.item_names[0]
        from app.engine.brief import canonical_item_for as _canon
        options = catalog_lookup(_canon(current_word))
        if options:
            furniture = design.get("furniture", [])
            if not isinstance(furniture, list):
                furniture = []

            # Identify the item actually being resized first, then use its own
            # catalog size as the baseline. Searching the catalog by the word
            # "desk" found "Study Desk 120cm" even when the design held the
            # 90cm kids table, so "bigger" resized from the wrong baseline and
            # could report no change while a larger option existed.
            target = None
            for f in furniture:
                if not isinstance(f, dict):
                    continue
                sku = str(f.get("sku", "")).strip()
                if sku and any(o.sku == sku for o in options):
                    target = f
                    break
            if target is None:
                for f in furniture:
                    if isinstance(f, dict) and current_word.lower() in str(
                        f.get("name", "")
                    ).lower():
                        target = f
                        break

            current_area: Optional[int] = None
            if target is not None:
                sku = str(target.get("sku", "")).strip()
                for o in options:
                    if o.sku == sku:
                        current_area = o.width_cm * o.depth_cm
                        break
            if current_area is None:
                for o in options:
                    if current_word.lower() in str(o.name).lower():
                        current_area = o.width_cm * o.depth_cm
                        break

            pool = [
                o for o in options
                if (o.width_cm * o.depth_cm) > (current_area or 0)
            ] if bigger else [
                o for o in options
                if current_area is not None
                and (o.width_cm * o.depth_cm) < current_area
            ]
            if pool:
                product = sorted(pool, key=lambda i: abs(
                    i.width_cm * i.depth_cm - (current_area or 0)
                ))[0]
                if target is not None:
                    target["name"] = product.name
                    target["sku"] = product.sku
                    target["category"] = product.category
                    target["brand_suggestions"] = [product.brand]
                    changed.append(f"resized to {product.name}")
            if not changed:
                if target is None:
                    # "Make the desk bigger" in a kitchen used to produce
                    # "Add chair only": the desk did not exist, so the
                    # resizer fell through to the catalog and changed an
                    # unrelated item. If the named thing is not in the design,
                    # say so and change nothing.
                    _refuse(design, current_word, catalog_lookup)
                else:
                    # Nothing in the catalog is larger (or smaller) than what
                    # is already there. Say so rather than reporting a resize
                    # that changed nothing, which reads as a broken button.
                    note = (
                        f"{current_word} is already the largest {current_word} in "
                        f"the catalog."
                        if bigger else
                        f"{current_word} is already the smallest {current_word} in "
                        f"the catalog."
                    )
                    design.setdefault("_refinement", {})["note"] = note

    elif intent.target == TARGET_LIGHTING:
        design["lighting_plan"] = (
            "Brighter layered scheme: a 4000K ceiling panel for even general "
            "light, a focused task lamp at the desk, and dimmable accent "
            "lighting. Higher lumen output than the previous plan."
        )
        changed.append("lighting_plan")

    if changed:
        metrics.increment("refinement.applied")
        logger.info("refinement_applied target=%s changed=%s", intent.target, changed)
    else:
        metrics.increment("refinement.no_op")

    design.setdefault("_refinement", {})["changed"] = changed
    return design


def _refuse(design: Dict[str, Any], word: str, catalog_lookup) -> None:
    """Record that the named item is not in this design, and change nothing.

    `changed` is left empty deliberately: the caller's diff then shows that
    nothing moved, and the note explains why. The message names the item in
    the client's own words.
    """
    from app.engine.brief import canonical_item_for as _canon

    name = word.strip().lower()
    available = []
    for entry in design.get("furniture") or []:
        if isinstance(entry, dict) and entry.get("name"):
            available.append(str(entry["name"]))
    design.setdefault("_refinement", {})["refused"] = True
    design["_refinement"]["message"] = f"There is no {name} in this design"
    design["_refinement"]["available"] = available
    metrics.increment("refinement.refused_missing_item")
    logger.info("refinement_refused item=%s canonical=%s", name, _canon(name))


def describe_change(intent: RefinementIntent) -> str:
    """A one-line human summary of what the refinement will touch."""
    if intent.target == TARGET_COLOR and intent.color_key:
        return f"Update the colour palette to {intent.color_key} only."
    if intent.target == TARGET_ITEM:
        action = intent.slots.get("action", "change")
        what = ", ".join(intent.item_names) or "the item"
        if action == "swap":
            return f"Swap {what} only, leaving the rest of the room unchanged."
        if action == "remove":
            return f"Remove {what} only."
        if action == "add":
            return f"Add {what} only."
        if action == "resize":
            return f"Resize {what} only, updating its catalog price."
    if intent.target == TARGET_LAYOUT:
        return "Re-arrange the specified items; the palette and costs for the rest stay."
    if intent.target == TARGET_LIGHTING:
        return "Rewrite the lighting plan only."
    if intent.target == TARGET_BUDGET:
        return "Re-select items to fit the budget, keeping the concept."
    return "I could not tell which part to change. Try naming the item or the aspect."
