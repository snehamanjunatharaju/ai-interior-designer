"""
Colour schemes, defined in code.

The client picks one scheme on the form. The palette used to be written by
the model from the brief, so a "Warm Earthy" request could come back with
navy, sky blue and teal - the model had read "neutral" or simply drifted. A
named scheme is a specification, not a creative decision, so the palette is
looked up here and installed on the design after generation.

Every palette carries a `forbidden` list of colour words that must not appear
in a design of that scheme. The validator checks both the named colours and
the hex codes against it, so the failure is caught even if the model writes
"a lazy afternoon glow" instead of naming a colour.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Canonical scheme name -> palette.
#
# `hex_codes` are validated by saturation in the validator, so the values are
# chosen deliberately: vibrant passes the 0.35 threshold on at least two
# colours, monochrome stays below 0.5 average, earthy stays below the vibrancy
# threshold while remaining clearly warm.
SCHEMES: Dict[str, Dict[str, Any]] = {
    "neutral": {
        "label": "Neutral Tones",
        "primary": "warm white",
        "secondary": "greige",
        "accent": "sage",
        "neutrals": ["white", "soft grey"],
        "hex_codes": ["#F2EFE9", "#CFC7BA", "#9CAF88"],
        "forbidden": [],
    },
    "warm": {
        # The form's "Warm Earthy". Kept as its own entry rather than folded
        # into `earthy` so the label the client saw is the label in the output.
        "label": "Warm Earthy",
        "primary": "terracotta",
        "secondary": "olive",
        "accent": "rust",
        "neutrals": ["sand", "off white"],
        "hex_codes": ["#C0704A", "#7A8B4F", "#A63D2F", "#E3D5C3"],
        "forbidden": [
            "navy", "sky blue", "teal", "cobalt", "royal blue", "turquoise",
            "cyan", "blue",
        ],
    },
    "earthy": {
        "label": "Earthy",
        "primary": "terracotta",
        "secondary": "olive",
        "accent": "rust",
        "neutrals": ["sand", "off white"],
        "hex_codes": ["#C0704A", "#7A8B4F", "#A63D2F", "#E3D5C3"],
        "forbidden": [
            "navy", "sky blue", "teal", "cobalt", "royal blue", "turquoise",
            "cyan", "blue",
        ],
    },
    "pastel": {
        "label": "Soft Pastels",
        "primary": "pale pink",
        "secondary": "powder blue",
        "accent": "mint",
        "neutrals": ["white"],
        "hex_codes": ["#F7C6D0", "#BBD7EE", "#BFEAD9"],
        "forbidden": ["black", "charcoal", "maroon"],
    },
    "monochrome": {
        "label": "Monochrome",
        "primary": "charcoal",
        "secondary": "grey",
        "accent": "silver",
        "neutrals": ["white", "black"],
        "hex_codes": ["#2E2E2E", "#7A7A7A", "#C9C9C9"],
        "forbidden": ["coral", "marigold", "terracotta", "teal", "magenta"],
    },
    "vibrant": {
        "label": "Vibrant",
        "primary": "coral",
        "secondary": "teal",
        "accent": "marigold",
        "neutrals": ["white", "pale grey"],
        "hex_codes": ["#FF6B5B", "#17A2A2", "#F2A93B"],
        "forbidden": [],
    },
    "cool": {
        "label": "Cool Tones",
        "primary": "teal",
        "secondary": "navy",
        "accent": "sky blue",
        "neutrals": ["white", "light grey"],
        "hex_codes": ["#17A2A2", "#1E3A5F", "#4A9FD8"],
        "forbidden": ["terracotta", "rust", "marigold", "coral"],
    },
    "earthy_green": {
        "label": "Earthy Greens",
        "primary": "olive",
        "secondary": "sage",
        "accent": "forest green",
        "neutrals": ["off white", "sand"],
        "hex_codes": ["#7A8B4F", "#A3B18A", "#4F7942"],
        "forbidden": ["navy", "coral", "magenta"],
    },
    "jewel": {
        "label": "Jewel Tones",
        "primary": "emerald",
        "secondary": "sapphire",
        "accent": "mustard",
        "neutrals": ["cream"],
        "hex_codes": ["#1F6F54", "#1E3A8A", "#C9A227"],
        "forbidden": ["beige", "pastel", "pale grey"],
    },
}

# Aliases accepted from the form or prose.
_ALIASES: Dict[str, str] = {
    "warm earthy": "warm",
    "warm": "warm",
    "warm neutral": "warm",
    "neutral tones": "neutral",
    "neutral": "neutral",
    "soft pastels": "pastel",
    "pastel": "pastel",
    "monochrome": "monochrome",
    "vibrant": "vibrant",
    "earthy": "earthy",
    "earthy green": "earthy_green",
    "cool": "cool",
    "cool tones": "cool",
    "jewel": "jewel",
    "jewel tones": "jewel",
}


def scheme_key(value: Optional[str]) -> Optional[str]:
    """Resolve any spelling of a scheme to its key in `SCHEMES`."""
    text = (value or "").strip().lower().replace("_", " ")
    if not text:
        return None
    if text in _ALIASES:
        return _ALIASES[text]
    # Containment rather than prefix, so "earthy green" resolves before the
    # bare "earthy" it contains, and "warm earthy tones" still resolves.
    for alias in sorted(_ALIASES, key=len, reverse=True):
        if alias in text:
            return _ALIASES[alias]
    return None


def palette_for(value: Optional[str]) -> Dict[str, Any]:
    """The palette for a scheme, or the neutral default for an unknown one.

    Returns a copy: the caller installs it on the design and may enrich it
    (variation titles, for instance) without mutating the shared constant.
    """
    key = scheme_key(value) or "neutral"
    palette = SCHEMES.get(key, SCHEMES["neutral"])
    return {
        "scheme": key,
        "label": palette["label"],
        "primary": palette["primary"],
        "secondary": palette["secondary"],
        "accent": palette["accent"],
        "neutrals": list(palette["neutrals"]),
        "hex_codes": list(palette["hex_codes"]),
        "forbidden": list(palette["forbidden"]),
    }


def forbidden_words(value: Optional[str]) -> List[str]:
    """Colour words a design of this scheme must not use."""
    return palette_for(value)["forbidden"]
