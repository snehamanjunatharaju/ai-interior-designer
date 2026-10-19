"""
Making the response agree with itself.

Four checks that share a shape: the client is shown several views of the same
design, and the views were able to disagree. Each one below was observed.

  * the colour chips rendered from `hex_codes` while the swatch labels came from
    `primary`/`secondary`/`accent`, so a card showed "Sage" next to a coral chip;
  * the pros read "the rug anchors the space" for a design with no rug bought;
  * the request listed a Bookshelf and the shopping list silently omitted it,
    with nothing failing;
  * a Strict-Vastu brief produced a compliance score with no bearing supplied.

`audit_option` is the single place these are checked, so the response cannot be
assembled from parts that disagree. It is deliberately deterministic - no model
is asked whether the design is self-consistent, because that is a question about
its own output and a model will answer "yes".
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

_HEX = re.compile(r"^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")

# Colour name -> the hue range (degrees) that actually reads as that colour.
#
# These ranges were set by measuring the real hex values the catalog and the
# palette generator produce, not by guessing. A guessed range made every
# palette report a mismatch - including correct ones - which would have made
# this check noise rather than signal. Measurements taken:
#   coral   #FF6B5B ->   6      sage    #A5B478 ->  75
#   marigold#E0A93B ->  40      teal    #17A2A2 -> 180
#   navy    #1B3B6F -> 217      slate   #2F4858 -> 203
#
# "neutral" has no hue band: it is matched on low saturation instead.
_FAMILY_HUE: Dict[str, Optional[tuple]] = {
    "red": (0, 12),
    "coral": (0, 20),
    "terracotta": (5, 28),
    "brown": (12, 32),
    "rust": (10, 30),
    "orange": (20, 33),
    "marigold": (33, 48),
    "mustard": (38, 50),
    "yellow": (45, 68),
    "beige": (28, 50),
    "cream": (28, 50),
    "gold": (38, 55),
    "olive": (52, 78),
    "sage": (60, 95),
    "green": (78, 160),
    "mint": (130, 170),
    "teal": (165, 195),
    "cyan": (180, 200),
    "slate": (195, 215),
    "steel": (200, 220),
    "blue": (205, 250),
    "navy": (210, 240),
    "denim": (205, 230),
    "indigo": (245, 275),
    "violet": (265, 290),
    "purple": (280, 310),
    "lavender": (250, 290),
    "plum": (295, 330),
    "pink": (320, 350),
    "rose": (335, 358),
    "magenta": (300, 330),
}


def _hex_to_rgb(value: str) -> Optional[tuple]:
    text = str(value or "").strip()
    if not text.startswith("#"):
        return None
    if len(text) == 4:
        text = "#" + "".join(c * 2 for c in text[1:])
    if len(text) != 7:
        return None
    try:
        return (int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16))
    except ValueError:
        return None


def _hue(rgb: tuple) -> float:
    import colorsys

    r, g, b = (c / 255 for c in rgb)
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return h * 360


def _saturation(rgb: tuple) -> float:
    import colorsys

    r, g, b = (c / 255 for c in rgb)
    _, s, _ = colorsys.rgb_to_hsv(r, g, b)
    return s


def normalise_hex(value: str) -> Optional[str]:
    """Uppercase, expand shorthand, or return None if it is not a colour."""
    text = str(value or "").strip()
    if not text.startswith("#"):
        text = "#" + text
    if len(text) == 4:
        text = "#" + "".join(c * 2 for c in text[1:])
    text = text.upper()
    return text if _HEX.match(text) else None


def palette_problems(palette: Any) -> List[Dict[str, str]]:
    """
    Does the palette describe itself consistently?

    Checks the three things that were observed going wrong:

      * a malformed hex, which the UI renders as an invisible or wrong chip;
      * a named colour with no corresponding hue anywhere in the hex list, so
        the label and the swatch are describing different colours;
      * a hex list that is empty while named colours are present, meaning the
        chips cannot be drawn at all.
    """
    if not isinstance(palette, dict):
        return []
    problems: List[Dict[str, str]] = []

    raw_hexes = [str(h) for h in (palette.get("hex_codes") or [])]
    hexes = [normalise_hex(h) for h in raw_hexes]
    invalid = [raw for raw, ok in zip(raw_hexes, hexes) if not ok]

    if invalid:
        problems.append({
            "code": "invalid_hex",
            "severity": "error",
            "message": (
                f"Palette contains values that are not colours: "
                f"{', '.join(invalid[:4])}. These render as blank chips."
            ),
            "remedy": "Use 6-digit hex values such as #2F4858.",
        })

    valid = [h for h in hexes if h]  # normalised, only real colours
    named = [
        str(palette.get(k, "")).strip()
        for k in ("primary", "secondary", "accent")
        if str(palette.get(k, "")).strip()
    ]

    if named and not hexes:
        problems.append({
            "code": "palette_without_hex",
            "severity": "error",
            "message": (
                f"The palette names colours ({', '.join(named)}) but supplies no "
                f"hex codes, so the colour chips cannot be drawn."
            ),
            "remedy": "Provide hex_codes alongside the colour names.",
        })
        return problems

    # A named colour that appears nowhere in the hex list is a mismatch between
    # the label and the swatch.
    if valid and named:
        rgbs = [r for r in (_hex_to_rgb(h) for h in valid) if r]
        hues = [_hue(r) for r in rgbs]
        sats = [_saturation(r) for r in rgbs]
        for label in named:
            family = _family_for(label)
            if family is None:
                continue
            if not _hue_matches(family, hues, sats):
                problems.append({
                    "code": "palette_name_hex_mismatch",
                    "severity": "warning",
                    "message": (
                        f"The palette names '{label}' but no hex code in the "
                        f"list is a {label} hue, so the label and the colour "
                        f"chips disagree."
                    ),
                    "remedy": (
                        "Either rename the colour to match the hex codes, or "
                        "add a hex code in that family."
                    ),
                })
                break

    return problems


def _family_for(label: str) -> Optional[str]:
    """Colour family for a name, or None when it is not a recognisable hue."""
    text = str(label or "").strip().lower()
    if not text:
        return None
    for family, range_ in _FAMILY_HUE.items():
        if family in text:
            return family
    if any(w in text for w in ("white", "cream", "ivory")):
        return "neutral"
    if any(w in text for w in ("black", "charcoal")):
        return "neutral"
    if any(w in text for w in ("grey", "gray")):
        return "neutral"
    return None


def _hue_matches(family: str, hues: Sequence[float], sats: Sequence[float]) -> bool:
    """
    Does any hex in the list actually read as this colour family?

    The tolerance is deliberately wide (15 degrees either side). A palette is
    not required to hit a textbook hue exactly, and a check that flags every
    legitimate palette teaches the reader to ignore it. It only needs to catch a
    genuinely different colour - "Coral" labelled beside a navy chip.
    """
    band = _FAMILY_HUE.get(family)
    if band is None:
        # Neutral names match on low saturation instead of hue.
        return any(s < 0.18 for s in sats)
    low, high = band
    for hue in hues:
        if low - 15 <= hue <= high + 15:
            return True
        if low - 15 < 0 and hue >= 345:   # wraps past 360
            return True
    return False


def requested_item_gaps(
    option: Dict[str, Any], requested: Sequence[str]
) -> List[Dict[str, str]]:
    """
    Requested items absent from this option's shopping list.

    `missing_requirements` reports a bare `item`/`reason`; it is reshaped here so
    every finding this module returns has the same keys, which is what lets the
    caller render them uniformly.
    """
    from app.engine.assumptions import missing_requirements

    gaps: List[Dict[str, str]] = []
    for gap in missing_requirements(option, requested):
        if gap.get("fatal") != "true":
            continue
        gaps.append({
            "code": "requested_item_missing",
            "severity": "error",
            "message": (
                f"The brief asked for a {gap['item']} but this option's shopping "
                f"list does not contain one."
            ),
            "remedy": (
                f"Add a real catalog {gap['item']} to the shopping list. The "
                f"brief is not satisfied by describing the item in prose."
            ),
        })
    return gaps


def ungrounded_claims(option: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Pros and cons that describe something the design does not contain.

    A pro is only useful if it is true of *this* design. "The rug anchors the
    seating" is filler on a design with no rug, and a client who notices stops
    trusting the rest of the card.
    """
    from app.engine.assumptions import bought_categories, mentioned_items

    bought = " ".join(bought_categories(option))
    problems: List[Dict[str, str]] = []

    for kind, words in mentioned_items(option).items():
        # Only judge the list fields, not the whole design: a mirror may
        # legitimately be "recommended" in decor_suggestions.
        for field in ("pros",):
            entries = option.get(field) or []
            for entry in entries:
                text = str(entry).lower()
                hits = [w for w in words if w in text]
                if hits and kind not in bought:
                    problems.append({
                        "code": "ungrounded_pro",
                        "severity": "warning",
                        "message": (
                            f"The {field} mention a {kind} ({', '.join(hits)}) "
                            f"but no {kind} item was bought."
                        ),
                        "remedy": (
                            "Add the item to the shopping list, or rewrite the "
                            "line so it describes something actually included."
                        ),
                    })
    return problems


def audit_option(
    option: Dict[str, Any],
    requested: Sequence[str] = (),
    strict_vastu: bool = False,
    has_bearings: bool = True,
) -> Dict[str, Any]:
    """
    Everything wrong with one option, in one place.

    Returns `problems` (each with a code, severity and remedy) and a `valid`
    flag computed from the error-severity problems only - a warning should not
    fail a design that is otherwise sound.
    """
    problems: List[Dict[str, str]] = []
    problems.extend(palette_problems(option.get("color_palette")))
    problems.extend(requested_item_gaps(option, requested))
    problems.extend(ungrounded_claims(option))

    if strict_vastu and not has_bearings:
        problems.append({
            "code": "vastu_claim_without_bearing",
            "severity": "error",
            "message": (
                "A Strict Vastu brief asked for a compliance assessment, but no "
                "door or window bearing was supplied, so the claim cannot be "
                "made."
            ),
            "remedy": (
                "Supply the door and window directions. Until then the Vastu "
                "score is withheld rather than guessed."
            ),
        })

    errors = [p for p in problems if p.get("severity") == "error"]
    return {
        "valid": not errors,
        "problems": problems,
        "error_count": len(errors),
        "warning_count": len(problems) - len(errors),
    }


def describe_problem(problem: Dict[str, str]) -> str:
    """One line for the UI."""
    return f"{problem.get('message', '')} {problem.get('remedy', '')}".strip()