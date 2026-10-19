"""
Who lives in this room, and what does that forbid.

Two requirements from the brief could not be enforced before this existed:

  * an adults-only brief was served a Kids Study Table and a Toy Storage Chest;
  * a Work-From-Home brief could come back with a desk and no chair, which is
    not a workspace.

Both are consequences of the same gap: the catalog had no idea who a product
was for, and the brief had no structured idea who the room was for. The catalog
now carries an `audience` column (`adult` / `kids` / `any`) and this module
reads the brief's occupants into a single audience decision that the optimizer,
the validator and the validator's exclusion list all consult.

The rule is deliberately conservative. "adults" wins over "kids" when a brief
mentions both - a family room still cannot have a bunk bed as its only bed -
and an unstated audience allows everything, because filtering on a guess would
reject legitimate designs.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

ADULT = "adult"
KIDS = "kids"
ANY = "any"

# Words in a brief that establish who the room is for.
_ADULT_TERMS = re.compile(
    r"\b(adult|adults|grown[- ]?ups?|parents?|professionals?|couple|"
    r"master|master bedroom|elderly|senior|seniors|tenant|tenants)\b",
    re.I,
)
_KIDS_TERMS = re.compile(
    r"\b(kid|kids|child|children|toddler|toddlers|baby|babies|infant|"
    r"teen|teens|teenager|student|students|school|playroom|nursery)\b",
    re.I,
)

# Work-from-home phrases. Any of these makes a desk and chair mandatory: a
# study desk with nowhere to sit is a shelf.
_WFH_TERMS = re.compile(
    r"\b(work from home|wfh|home office|home[- ]office|remote work|"
    r"remote[- ]working|workstation|covid[- ]?desk|telecommute)\b",
    re.I,
)


def infer_audience(
    occupants: Optional[Iterable[str]] = None,
    free_text: str = "",
) -> str:
    """
    Decide whether this room serves adults, kids, or is unstated.

    Adults win a tie. A room described as "for adults and guests" should not
    receive a toddler bed, and a "parents' room" is an adult room even when
    children are mentioned in passing.
    """
    blob = " ".join([*(occupants or []), free_text or ""])

    has_adults = bool(_ADULT_TERMS.search(blob))
    has_kids = bool(_KIDS_TERMS.search(blob))

    if has_adults and has_kids:
        return ADULT
    if has_kids:
        return KIDS
    if has_adults:
        return ADULT
    return ANY


def requires_wfh_kit(
    occupants: Optional[Iterable[str]] = None,
    free_text: str = "",
    room_type: str = "",
) -> bool:
    """True when the brief describes a work-from-home setup."""
    blob = " ".join([*(occupants or []), free_text or ""])
    if _WFH_TERMS.search(blob):
        return True
    return str(room_type or "").lower() == "home_office"


def product_is_allowed(product, audience: str) -> bool:
    """
    May this product appear in a room for `audience`?

    "any" products are always allowed. A kids' product is never allowed in an
    adults-only room.
    """
    if audience == ANY:
        return True
    product_audience = getattr(product, "audience", ANY)
    if product_audience == ANY:
        return True
    return product_audience == audience


def filter_products(products: Sequence, audience: str) -> List:
    """Drop every product the audience forbids."""
    return [p for p in products if product_is_allowed(p, audience)]


def disallowed_products(products: Sequence, audience: str) -> List:
    """The complement of `filter_products`, used for validator messages."""
    return [p for p in products if not product_is_allowed(p, audience)]


def audience_rationale(audience: str) -> str:
    """One line explaining an audience decision, for the assumptions panel."""
    return {
        ADULT: "Read as an adults-only room, so children's furniture is excluded.",
        KIDS: "Read as a children's room.",
        ANY: "Occupants were not specified, so no audience filtering was applied.",
    }.get(audience, "Occupants were not specified.")


def required_categories_for(
    audience: str,
    wfh: bool,
) -> Dict[str, int]:
    """
    Categories the brief obliges the design to contain.

    A work-from-home brief gains a hard requirement of one desk and one chair.
    This is separate from whatever the client typed, so "I work from home"
    without the words "desk" or "chair" still produces both.
    """
    required: Dict[str, int] = {}
    if wfh:
        required["desk"] = max(1, required.get("desk", 0))
        required["chair"] = max(1, required.get("chair", 0))
    return required