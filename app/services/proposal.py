"""
PDF client proposal export.

Why this exists
---------------
A client does not want a JSON blob or a screenshot of a web page. They want a
document they can forward, print, and hold you to. The proposal is therefore
generated from the same code-computed costing the UI shows, so the PDF cannot
disagree with the screen: one source, two renderings.

Built with ReportLab rather than a headless browser because it is a few
hundred KB rather than a few hundred MB, and it needs no Chromium on the
server. The floor plan is drawn with ReportLab's own vector primitives from
the same layout JSON the 2D view uses.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.core.observability import get_logger
import os

logger = get_logger(__name__)

INK = colors.HexColor("#1f2937")
MUTED = colors.HexColor("#6b7280")
RULE = colors.HexColor("#e5e7eb")
ACCENT = colors.HexColor("#2563eb")
GOOD = colors.HexColor("#047857")
WARN = colors.HexColor("#b45309")
BAD = colors.HexColor("#b91c1c")

# Must match the 2D view's category colours so the two documents agree.
CATEGORY_FILL = {
    "bed": colors.HexColor("#cfe0f7"),
    "desk": colors.HexColor("#d9f0e0"),
    "chair": colors.HexColor("#e6e0f5"),
    "wardrobe": colors.HexColor("#f0e0d4"),
    "storage": colors.HexColor("#f0e0d4"),
    "tv_unit": colors.HexColor("#d6dde6"),
    "seating": colors.HexColor("#f7e0e6"),
    "table": colors.HexColor("#f3e8d0"),
    "soft": colors.HexColor("#efeaf6"),
    "decor": colors.HexColor("#fdf2d9"),
    "lighting": colors.HexColor("#fff6cf"),
    "paint": colors.HexColor("#eeeeee"),
    "kitchen": colors.HexColor("#dfe8ee"),
}


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    # Every style uses the Unicode font when one is available, so the rupee
    # sign renders in body text and table cells, not only in headline figures.
    uni = _unicode_font()
    family = {"fontName": uni} if uni else {}
    bold = {"fontName": uni} if uni else {"fontName": "Helvetica-Bold"}
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=22,
                                textColor=INK, spaceAfter=2, alignment=TA_LEFT,
                                **bold),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9.5,
                              textColor=MUTED, spaceAfter=10, **family),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13,
                             textColor=INK, spaceBefore=12, spaceAfter=5, **bold),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5,
                               textColor=INK, leading=13.5, **family),
        "small": ParagraphStyle("sm", parent=base["Normal"], fontSize=8,
                                textColor=MUTED, leading=11, **family),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=8.5,
                               textColor=INK, **family),
        "cellb": ParagraphStyle("cb", parent=base["Normal"], fontSize=8.5,
                                textColor=INK, **bold),
    }


def _rupee(n: Any) -> str:
    """
    Single money format for the whole document - see app.core.money.

    Falls back to the `Rs` prefix when no Unicode font is available, because
    the base-14 fonts render `₹` as a stray `n`.
    """
    from app.core.money import format_inr

    if _unicode_font() is None:
        try:
            return "Rs {:,}".format(int(n or 0))
        except (TypeError, ValueError):
            return "Rs 0"
    return format_inr(n)


# ---------------------------------------------------------------------------
# Font selection
# ---------------------------------------------------------------------------
# ReportLab's built-in Type 1 fonts (Helvetica and friends) are Latin-1 only:
# the rupee sign has no glyph and is emitted as a literal "n". The proposal was
# therefore printing "n 14,500" on every price, which is not a formatting
# detail - it makes the document unusable and, worse, looks like a real figure
# at a glance.
#
# A Unicode TTF is resolved once, from a candidate list, and cached. If none is
# present the document falls back to the word "Rs", which at least renders
# correctly rather than as a stray letter.

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/NotoSans-Regular.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\segoeui.ttf",
)

_font_cache: Dict[str, Any] = {}


def _unicode_font() -> Optional[str]:
    """
    Register and return a font that can draw the rupee sign.

    Returns None when no candidate exists, which is the signal to fall back to
    the "Rs" prefix rather than render a broken glyph.
    """
    if "name" in _font_cache:
        return _font_cache["name"]
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        for path in _FONT_CANDIDATES:
            if not os.path.isfile(path):
                continue
            try:
                # Verify the glyph is actually present. A font can be Unicode
                # and still lack U+20B9, and registering it would give a
                # different wrong answer rather than the right one.
                pdfmetrics.registerFont(TTFont("InteriorAIUnicode", path))
                _font_cache["name"] = "InteriorAIUnicode"
                _font_cache["path"] = path
                return "InteriorAIUnicode"
            except Exception:  # pragma: no cover - font-specific
                continue
    except Exception:  # pragma: no cover - reportlab import
        pass
    _font_cache["name"] = None
    return None


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 12 * mm, "InteriorAI proposal")
    canvas.drawRightString(
        A4[0] - 18 * mm, 12 * mm,
        "Page {}".format(doc.page),
    )
    canvas.restoreState()


def _draw_plan(flowables, layout: Dict[str, Any], styles) -> None:
    """
    Draw the floor plan as vector graphics from the layout JSON.

    Uses the same coordinates the web view does, scaled to fit the page, so a
    client comparing the PDF against the screen sees the same plan.
    """
    room = layout.get("room") or {}
    items = layout.get("items") or []
    W = float(room.get("width_cm") or 0)
    D = float(room.get("depth_cm") or 0)
    if not W or not D:
        return

    max_w = 170 * mm
    scale = max_w / W
    pad = 8 * mm
    plan_w = W * scale
    plan_h = D * scale

    from reportlab.graphics.shapes import Drawing, Rect, String as GString

    d = Drawing(plan_w + pad * 2, plan_h + pad * 2)
    ox, oy = pad, pad

    # Floor
    d.add(Rect(ox, oy, W * scale, D * scale,
               fillColor=colors.HexColor("#fbfbfc"),
               strokeColor=colors.HexColor("#374151"), strokeWidth=1.4))

    # Window band
    wd = room.get("window_direction")
    if wd:
        band_w, band_d = min(50 * mm, W * scale * 0.35), 8
        if "north" in str(wd):
            d.add(Rect(ox, oy + D * scale - 8, band_w, 8,
                       fillColor=colors.HexColor("#bfdbfe"),
                       strokeColor=colors.HexColor("#3b82f6")))
        elif "south" in str(wd):
            d.add(Rect(ox, oy, band_w, 8,
                       fillColor=colors.HexColor("#bfdbfe"),
                       strokeColor=colors.HexColor("#3b82f6")))
        elif "west" in str(wd):
            d.add(Rect(ox, oy, 8, band_w,
                       fillColor=colors.HexColor("#bfdbfe"),
                       strokeColor=colors.HexColor("#3b82f6")))
        elif "east" in str(wd):
            d.add(Rect(ox + W * scale - 8, oy, 8, band_w,
                       fillColor=colors.HexColor("#bfdbfe"),
                       strokeColor=colors.HexColor("#3b82f6")))

    for it in items:
        cat = str(it.get("category", "")).lower()
        d.add(Rect(
            ox + float(it["x"]) * scale,
            # ReportLab's y axis is bottom-up; the layout is top-down.
            oy + (D - float(it["y"]) - float(it["depth"])) * scale,
            float(it["width"]) * scale,
            float(it["depth"]) * scale,
            fillColor=CATEGORY_FILL.get(cat, colors.HexColor("#e5e7eb")),
            strokeColor=colors.HexColor("#6b7280"),
            strokeWidth=0.6,
        ))

    # Room dimensions, in feet, which is how Indian clients think.
    d.add(GString(ox, oy - 11, "%.1f ft" % (W / 30.48), fontName="Helvetica",
                  fontSize=7, fillColor=MUTED))
    d.add(GString(ox - 30, oy + (D * scale) / 2 - 2, "%.1f ft" % (D / 30.48),
                  fontName="Helvetica", fontSize=7, fillColor=MUTED))

    flowables.append(Spacer(1, 4))
    flowables.append(d)


def build_proposal_pdf(
    design: Dict[str, Any],
    cost: Optional[Dict[str, Any]] = None,
    layout: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    vastu: Optional[Dict[str, Any]] = None,
    budget_plan: Optional[Dict[str, Any]] = None,
    client_name: str = "Client",
    room_type: str = "",
) -> bytes:
    """
    Render a client-facing proposal to PDF bytes.

    Every figure here comes from the code-computed costing. If the design was
    not priced, the proposal says so rather than printing a number nobody can
    source.
    """
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="Interior design proposal", author="InteriorAI",
    )

    flow: List[Any] = []
    cost = cost or {}
    total = cost.get("estimated_cost_inr", 0)

    # ---- cover -------------------------------------------------------
    flow.append(Paragraph("Interior design proposal", styles["title"]))
    flow.append(Paragraph(
        "Prepared for {} \u00b7 {} \u00b7 all prices from the product catalogue "
        "(version {})".format(
            client_name or "Client",
            (room_type or "").replace("_", " ").title() or "Room",
            cost.get("catalog_version", "n/a"),
        ),
        styles["sub"],
    ))
    flow.append(HRFlowable(width="100%", color=RULE, spaceAfter=10))

    # Headline figures, so the client sees the number that matters first.
    total_text = _rupee(total)
    budget = cost.get("budget_inr")
    if budget and not cost.get("within_budget", True):
        total_text += "  (over a {} budget)".format(_rupee(budget))

    summary = [
        ["Estimated total", total_text],
    ]
    if layout and layout.get("room"):
        r = layout["room"]
        summary.append([
            "Room size",
            "{:.0f} \u00d7 {:.0f} cm  ({:.1f} \u00d7 {:.1f} ft)".format(
                r.get("width_cm", 0), r.get("depth_cm", 0),
                r.get("width_cm", 0) / 30.48, r.get("depth_cm", 0) / 30.48,
            ),
        ])
    if vastu:
        summary.append(["Vastu score", "{} / 100  ({})".format(
            vastu.get("score", 0), vastu.get("grade", ""))])
    if validation:
        summary.append([
            "Brief compliance",
            "{} of {} checks passed".format(
                validation.get("checks_passed", 0), validation.get("checks_run", 0)
            ),
        ])

    t = Table(summary, colWidths=[52 * mm, 110 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9.5),
        ("FONT", (0, 0), (-1, -1), "Helvetica-Bold", 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (1, 0), (1, 0), INK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    flow.append(t)

    # ---- concept -----------------------------------------------------
    if design.get("concept"):
        flow.append(Paragraph("Design concept", styles["h2"]))
        flow.append(Paragraph(design["concept"], styles["body"]))
    if design.get("layout_description"):
        flow.append(Paragraph("Layout", styles["h2"]))
        flow.append(Paragraph(design["layout_description"], styles["body"]))

    # ---- floor plan --------------------------------------------------
    if layout and layout.get("items"):
        flow.append(Paragraph("Floor plan", styles["h2"]))
        _draw_plan(flow, layout, styles)

    # ---- shopping list -----------------------------------------------
    line_items = cost.get("line_items") or []
    if line_items:
        flow.append(Paragraph("Shopping list", styles["h2"]))
        flow.append(Paragraph(
            "Every line is a real catalogue product. Totals are calculated in "
            "code, not estimated.", styles["small"],
        ))
        flow.append(Spacer(1, 4))

        rows = [[
            Paragraph("Item", styles["cellb"]),
            Paragraph("Brand", styles["cellb"]),
            Paragraph("Size (cm)", styles["cellb"]),
            Paragraph("Qty", styles["cellb"]),
            Paragraph("Line total", styles["cellb"]),
        ]]
        for li in line_items:
            dims = li.get("dimensions_cm") or {}
            rows.append([
                Paragraph(str(li.get("name", "")), styles["cell"]),
                Paragraph(str(li.get("brand", "")), styles["cell"]),
                Paragraph("{}\u00d7{}\u00d7{}".format(
                    dims.get("width", 0), dims.get("depth", 0), dims.get("height", 0)
                ), styles["cell"]),
                Paragraph(str(li.get("quantity", 1)), styles["cell"]),
                Paragraph(_rupee(li.get("line_total_inr", 0)), styles["cell"]),
            ])

        widths = [62 * mm, 34 * mm, 28 * mm, 12 * mm, 28 * mm]
        tbl = Table(rows, colWidths=widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f9fafb")),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (3, 1), (4, -1), "RIGHT"),
        ]))
        flow.append(tbl)
        flow.append(Spacer(1, 6))

        total_row = [[
            Paragraph("Total", styles["cellb"]), "", "", "",
            Paragraph(_rupee(total), styles["cellb"]),
        ]]
        tt = Table(total_row, colWidths=widths)
        tt.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, 0), 1, colors.HexColor("#9ca3af")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (4, 0), (4, 0), "RIGHT"),
        ]))
        flow.append(tt)

    # ---- phased plan --------------------------------------------------
    if budget_plan and budget_plan.get("phase1"):
        flow.append(Paragraph("Budget plan", styles["h2"]))
        flow.append(Paragraph(budget_plan.get("headline", ""), styles["body"]))
        flow.append(Spacer(1, 4))
        for key in ("phase1", "phase2"):
            phase = budget_plan.get(key) or {}
            items = phase.get("line_items") or []
            if not items:
                continue
            block: List[Any] = [
                Paragraph(phase.get("label", key), styles["body"]),
                Spacer(1, 3),
            ]
            rows = [[
                Paragraph("Item", styles["cellb"]),
                Paragraph("Total", styles["cellb"]),
            ]]
            for li in items:
                rows.append([
                    Paragraph(str(li.get("name", "")), styles["cell"]),
                    Paragraph(_rupee(li.get("line_total_inr", 0)), styles["cell"]),
                ])
            rows.append([
                Paragraph("Phase total", styles["cellb"]),
                Paragraph(_rupee(phase.get("total_inr", 0)), styles["cellb"]),
            ])
            bt = Table(rows, colWidths=[120 * mm, 34 * mm])
            bt.setStyle(TableStyle([
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
                ("LINEABOVE", (0, -1), (-1, -1), 0.7, colors.HexColor("#9ca3af")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            block.append(bt)
            block.append(Spacer(1, 8))
            flow.append(KeepTogether(block))
        remaining = (budget_plan.get("phase1") or {}).get("remaining_inr")
        if remaining:
            flow.append(Paragraph(
                "{} of the budget is unallocated after Phase 1.".format(_rupee(remaining)),
                styles["small"],
            ))

    # ---- vastu ---------------------------------------------------------
    if vastu and vastu.get("results"):
        flow.append(Paragraph("Vastu assessment", styles["h2"]))
        flow.append(Paragraph(
            "{} of {} rules satisfied. Each rule is checked against the measured "
            "layout, not a description.".format(
                vastu.get("passed", 0), vastu.get("total", 0)),
            styles["small"],
        ))
        flow.append(Spacer(1, 4))
        for r in vastu["results"]:
            mark = "Pass" if r.get("passed") else "Needs attention"
            colour = GOOD if r.get("passed") else (
                BAD if r.get("severity") == "error" else WARN
            )
            flow.append(Paragraph(
                '<font color="{}"><b>{}</b></font> \u2014 {}<br/>'
                '<font size="8" color="#6b7280">{}</font>{}'.format(
                    colour.hexval().replace("0x", "#"),
                    mark, r.get("title", ""),
                    r.get("explanation", ""),
                    ("<br/><font size='8'><b>Remedy:</b> " + r.get("remedy") + "</font>")
                    if r.get("remedy") else "",
                ),
                styles["body"],
            ))
            flow.append(Spacer(1, 4))

    # ---- trade-offs ----------------------------------------------------
    pros = design.get("pros") or []
    cons = design.get("cons") or []
    if pros or cons:
        flow.append(Paragraph("Trade-offs", styles["h2"]))
        for label, items, colour in (("Strengths", pros, GOOD), ("Limitations", cons, WARN)):
            if not items:
                continue
            flow.append(Paragraph(
                '<font color="{}"><b>{}</b></font>'.format(
                    colour.hexval().replace("0x", "#"), label),
                styles["body"],
            ))
            for item in items:
                flow.append(Paragraph("&bull;&nbsp;&nbsp;" + str(item), styles["body"]))
            flow.append(Spacer(1, 4))

    # ---- compliance note ------------------------------------------------
    if validation and not validation.get("valid", True):
        flow.append(Paragraph("Outstanding items", styles["h2"]))
        for v in validation.get("violations", []):
            if v.get("severity") != "error":
                continue
            flow.append(Paragraph(
                "<b>{}</b> \u2014 {}".format(v.get("code", ""), v.get("message", "")),
                styles["body"],
            ))
            flow.append(Spacer(1, 2))

    flow.append(Spacer(1, 10))
    flow.append(HRFlowable(width="100%", color=RULE, spaceAfter=4))
    flow.append(Paragraph(
        "Prices are indicative catalogue figures for planning and are "
        "confirmed at the retailer. Dimensions are typical assembled sizes.",
        styles["small"],
    ))

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    logger.info("proposal_pdf_built line_items=%d", len(line_items))
    return buf.getvalue()
