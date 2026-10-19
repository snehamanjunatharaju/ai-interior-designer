"""
Money formatting, in one place.

This exists because the same amount was rendered three different ways across
the product: "Rs 4,100" in the PDF, "₹4100" in the API, and "Rs. 4,100" in
some validator messages. A client comparing the screen to the proposal could
reasonably think they were different numbers.

One rule, applied everywhere: the rupee sign, then Indian digit grouping
(lakh/crore grouping, not thousands), then nothing. No decimals, because every
amount in this system is a whole rupee figure read from the catalog.

Indian digit grouping is not a cosmetic detail either. "1,23,400" is Rs 123,400;
rendered with Western grouping as "123,400" it still reads as one hundred
twenty-three thousand to a reader who grew up on the other convention, which
is most of the audience for this product.
"""

from __future__ import annotations

from typing import Any

RUPEE = "₹"


def format_inr(amount: Any, *, with_symbol: bool = True) -> str:
    """
    Format a rupee amount with Indian digit grouping.

    >>> format_inr(4100)
    '₹4,100'
    >>> format_inr(123456)
    '₹1,23,456'
    >>> format_inr(None)
    '₹0'
    """
    try:
        value = int(round(float(amount or 0)))
    except (TypeError, ValueError):
        value = 0
    grouped = indian_grouping(abs(value))
    sign = "-" if value < 0 else ""
    prefix = RUPEE if with_symbol else ""
    return f"{sign}{prefix}{grouped}"


def format_inr_words(amount: Any) -> str:
    """
    Indian short scale: lakh and crore.

    Used where a rounded figure reads better than a long digit string, e.g.
    "about Rs 12.4 lakh" in a summary line. Exact figures always use
    `format_inr`, so this is never the only place a number appears.
    """
    try:
        value = float(amount or 0)
    except (TypeError, ValueError):
        return format_inr(0)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_00_00_000:
        return f"{sign}{RUPEE}{value / 1_00_00_000:.2f} crore"
    if value >= 1_00_000:
        return f"{sign}{RUPEE}{value / 1_00_000:.1f} lakh"
    return format_inr(-value if sign else value)


def indian_grouping(value: int) -> str:
    """
    Group digits as 1,23,456 rather than 123,456.

    The last three digits are always one group; everything before them is
    grouped in pairs. That is the whole rule.
    """
    digits = str(value)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    parts: list[str] = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])