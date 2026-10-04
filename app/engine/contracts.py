"""
The contract the Architect must satisfy, expressed as a Pydantic model.

The architect is a language model, so it will occasionally invent a SKU
("CHR-01"), invent a price, or invent a product that does not exist. Every
previous defence against that was a validation message afterwards - by then the
model had already written the item, and code had to clean up after it.

This module makes the invalid shapes *unrepresentable* instead. The architect
returns a `FurnitureSelection`: a catalog SKU and a quantity. Nothing else. It
cannot carry a price because there is no field for one, and it cannot carry a
made-up product because the SKU is checked against the catalog as the document
is parsed, so an unknown code fails validation at the boundary rather than
silently reaching the validator.

The distinction that matters: this model validates *selection*, not *arithmetic*.
Price, total and budget still come from `pricing.py` reading SQLite. This file
only decides which real products were chosen.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.observability import get_logger

logger = get_logger(__name__)


class FurnitureSelection(BaseModel):
    """
    One item the architect chose, as a catalog reference.

    `sku` must exist in the catalog. `quantity` is a positive integer. There is
    deliberately no price field: the model does not get to state one, so it
    cannot state a wrong one.
    """

    model_config = ConfigDict(extra="ignore")

    sku: str = Field(
        ...,
        min_length=1,
        description="A catalog SKU, copied verbatim. Never invented.",
    )
    quantity: int = Field(default=1, ge=1, le=20)

    @field_validator("sku")
    @classmethod
    def _sku_must_exist(cls, v: str) -> str:
        """Reject a SKU that is not in the catalog, at parse time."""
        sku = v.strip().upper()
        from app.catalog import catalog

        if catalog.get(sku) is None:
            raise ValueError(
                f"SKU '{v}' is not in the catalog. Choose a SKU from the catalog "
                f"list in the prompt, or omit the item entirely."
            )
        return sku


class PaletteSelection(BaseModel):
    """Colour palette as written by the model, validated for hex shape only."""

    model_config = ConfigDict(extra="ignore")

    primary: str = ""
    secondary: str = ""
    accent: str = ""
    neutrals: List[str] = Field(default_factory=list)
    hex_codes: List[str] = Field(default_factory=list)

    @field_validator("hex_codes")
    @classmethod
    def _hexes_must_be_hex(cls, v: List[str]) -> List[str]:
        """Drop anything that is not a hex colour rather than trusting it.

        A malformed hex is not a cosmetic problem: the UI paints these values
        directly as swatches, and a non-hex value renders as an invisible or
        wrongly-coloured chip.
        """
        cleaned = []
        for code in v:
            text = str(code).strip()
            if not text.startswith("#"):
                text = "#" + text
            text = text.upper()
            if len(text) == 4:  # #abc -> #AABBCC
                text = "#" + "".join(c * 2 for c in text[1:])
            if len(text) == 7 and all(c in "0123456789ABCDEF" for c in text[1:]):
                cleaned.append(text)
        return cleaned


class ArchitectProposal(BaseModel):
    """
    The structured subset of a design that the pipeline trusts.

    Prose fields are kept because they are the designer's actual contribution.
    Furniture is `List[FurnitureSelection]`, so a design that names an
    imaginary product is rejected as a whole and repaired in code rather than
    being passed down and half-priced.
    """

    model_config = ConfigDict(extra="ignore")

    title: str = Field(default="Design Option", max_length=200)
    concept: str = Field(default="", max_length=2000)
    layout_description: str = Field(default="", max_length=2000)
    zones: List[str] = Field(default_factory=list)
    furniture: List[FurnitureSelection] = Field(default_factory=list)
    color_palette: Optional[PaletteSelection] = None
    lighting_plan: str = Field(default="", max_length=1000)
    storage_strategy: str = Field(default="", max_length=1000)
    decor_suggestions: List[str] = Field(default_factory=list)
    vastu_notes: List[str] = Field(default_factory=list)
    brands: List[str] = Field(default_factory=list)
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)


class SelectionValidationError(ValueError):
    """Raised when an architect proposal cannot be trusted as written."""

    def __init__(self, problems: List[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems) or "invalid selection")


def validate_proposal(
    payload: Dict[str, Any],
) -> tuple[Optional[ArchitectProposal], List[str]]:
    """
    Validate an architect response against the catalog.

    Returns `(proposal, problems)`. When `proposal` is None the payload named
    products that do not exist and the caller must repair it in code - it must
    not pass the raw payload downstream, because a half-valid design prices
    against products that cannot be bought.

    Unknown *fields* are ignored rather than rejected: the model writes prose we
    do not model, and refusing the whole design over an extra key would be a
    regression. Only an item that cannot be sourced is fatal.
    """
    if not isinstance(payload, dict):
        return None, ["architect response was not a JSON object"]

    # Pull the selection apart by hand first. A Pydantic error on the whole
    # document does not say which SKU was wrong, and the repair path needs to
    # know that.
    raw_items = payload.get("furniture")
    if raw_items is not None and not isinstance(raw_items, list):
        return None, ["`furniture` was not a list"]

    problems: List[str] = []
    rejected: List[Dict[str, Any]] = []

    for entry in raw_items or []:
        if not isinstance(entry, dict):
            problems.append("a furniture entry was not an object")
            rejected.append({})
            continue
        sku = str(entry.get("sku", "")).strip().upper()
        try:
            qty = int(float(entry.get("quantity", 1) or 1))
        except (TypeError, ValueError):
            qty = 1
        if not sku:
            problems.append(
                f"item '{entry.get('name', 'unnamed')}' was chosen with no SKU; "
                f"every item must come from the catalog"
            )
            rejected.append(entry)
            continue
        try:
            FurnitureSelection(sku=sku, quantity=max(1, qty))
        except ValidationError as exc:
            problems.extend(_messages(exc))
            rejected.append(entry)

    # Rebuild with only the trustworthy selections, then let Pydantic check the
    # rest (palette hexes, string limits).
    candidate = dict(payload)
    candidate["furniture"] = [
        {"sku": str(e.get("sku", "")).strip().upper(),
         "quantity": max(1, int(float(e.get("quantity", 1) or 1)))}
        for e in (raw_items or [])
        if isinstance(e, dict)
        and str(e.get("sku", "")).strip()
        and _sku_is_real(str(e.get("sku", "")).strip().upper())
    ]

    try:
        proposal = ArchitectProposal(**candidate)
    except ValidationError as exc:
        problems.extend(_messages(exc))
        return None, problems

    if problems:
        return None, problems
    return proposal, []


def _sku_is_real(sku: str) -> bool:
    from app.catalog import catalog

    return catalog.get(sku) is not None


def _messages(exc: ValidationError) -> List[str]:
    """Flatten a Pydantic error into short, actionable sentences."""
    out: List[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "invalid")
        out.append(f"{loc}: {msg}" if loc else msg)
    return out


def selections_to_furniture(
    proposal: ArchitectProposal,
) -> List[Dict[str, Any]]:
    """
    Expand validated selections into design line items.

    Names, brands and categories come from the catalog, never from the model, so
    a line item cannot disagree with the product it points at.
    """
    from app.catalog import catalog

    items: List[Dict[str, Any]] = []
    for sel in proposal.furniture:
        product = catalog.get(sel.sku)
        if product is None:  # pragma: no cover - guarded by the schema
            continue
        items.append({
            "name": product.name,
            "sku": product.sku,
            "category": product.category,
            "quantity": sel.quantity,
            "brand_suggestions": [product.brand],
            "width_cm": product.width_cm,
            "depth_cm": product.depth_cm,
            "height_cm": product.height_cm,
        })
    return items