"""
Domain knowledge base for Indian interior design.

This is the corpus the RAG retriever grounds generations in. It is
deliberately specific to the Indian market (INR pricing, local brands,
Vastu, real-estate constraints) because that is where a generic model
produces the least useful output.

Each entry is chunked and embedded at startup.
"""

from __future__ import annotations

from typing import Dict, List

KNOWLEDGE_BASE: List[Dict[str, str]] = [
    # ---- Space planning -------------------------------------------------
    {
        "category": "space_planning",
        "source": "space_planning",
        "content": (
            "Small Indian living rooms (120-200 sq ft) should use wall-mounted floating "
            "shelves instead of floor units, a compact 2/3-seater sofa under 72 inches, "
            "and nesting coffee tables. Maintain a 90 cm (3 ft) circulation path between "
            "sofa and TV unit. Avoid deep furniture; standard Indian door widths are 90 cm "
            "so anything deeper than 60 cm blocks movement."
        ),
    },
    {
        "category": "space_planning",
        "source": "space_planning",
        "content": (
            "For Indian 2BHK and 3BHK apartments, a semi-open kitchen is standard: a "
            "counter-height breakfast bar or a glass partition separates cooking from the "
            "living area while keeping visual airflow. This raises perceived space and "
            "allows the kitchen to serve as a dining surface for 2-3 people."
        ),
    },
    {
        "category": "space_planning",
        "source": "space_planning",
        "content": (
            "Vastu Shastra principles for Indian homes: place the main door on the north, "
            "east, or north-east; keep the kitchen in the south-east with the stove facing "
            "east; avoid placing the master bedroom in the south-west; place the bed in "
            "the south-west corner with the headboard against the south or west wall; "
            "keep the bathroom in the north-west or west. Never place a mirror facing "
            "the bed."
        ),
    },
    # ---- Color ----------------------------------------------------------
    {
        "category": "color",
        "source": "color",
        "content": (
            "Indian interiors perform best with warm neutrals paired with a single saturated "
            "accent: ivory, beige, warm grey, or terracotta as base with teal, mustard, "
            "rust, or deep emerald as the accent. Use the 60-30-10 rule: 60% base, 30% "
            "secondary, 10% accent. Avoid all-white walls which show dust and stains in "
            "Indian climates, and avoid cool grey undertones in low-light rooms."
        ),
    },
    {
        "category": "color",
        "source": "color",
        "content": (
            "For small Indian rooms, paint the ceiling a shade lighter than the walls and "
            "use a single accent wall behind the sofa or bed in a deeper tone. This "
            "visually raises ceiling height and creates depth. Matte finishes suit dry "
            "climates; satin or eggshell in humid coastal cities like Chennai or Mumbai "
            "for washability."
        ),
    },
    # ---- Lighting -------------------------------------------------------
    {
        "category": "lighting",
        "source": "lighting",
        "content": (
            "Layered lighting for Indian homes: ambient ceiling fixtures at 2700-3000K warm "
            "white, task lighting under kitchen cabinets and over the study desk, and "
            "accent lighting with floor lamps or cove lighting. In monsoon-prone regions, "
            "budget for good drainage and sealed fixtures. Maximum LED wattage in a 150 sq "
            "ft room is roughly 150-200W total for even illumination."
        ),
    },
    {
        "category": "lighting",
        "source": "lighting",
        "content": (
            "Indian living rooms typically have one main window facing either west (hot "
            "afternoon sun) or south. For west-facing windows use blackout curtains plus "
            "a sheer layer, and position the sofa perpendicular to the window to avoid "
            "glare on a TV. For east-facing, sheer cotton diffuses morning light well."
        ),
    },
    # ---- Materials & budget ---------------------------------------------
    {
        "category": "materials",
        "source": "materials",
        "content": (
            "Budget breakdown for an Indian 3BHK mid-range renovation (INR 5-15 lakh): "
            "modular kitchen 25-30%, furniture 20-25%, civil and paint 15%, false ceiling "
            "and lighting 10%, flooring 10%, soft furnishings and decor 8%. Marble and "
            "Italian tiles can triple flooring cost; vitrified tiles offer 60-70% of the "
            "look at a fraction of the price."
        ),
    },
    {
        "category": "materials",
        "source": "materials",
        "content": (
            "Indian market furniture brands by segment: budget (under Rs 15,000 per piece) "
            "Nilkamal, Godrej Interio, Urban Ladder basics, Pepperfry, Wakefit. Mid-range "
            "(Rs 15,000-50,000) Urban Ladder, Pepperfry, Souk, Chumbak. Premium (above Rs "
            "50,000) Dinesh, Aesthete, Roohi, Studio Mudhaliar, Godrej Interio premium "
            "lines. Always verify ply and board certifications before purchase."
        ),
    },
    {
        "category": "materials",
        "source": "materials",
        "content": (
            "Modular kitchen pricing in India: a basic 10 sq ft kitchen starts around Rs "
            "80,000 with laminate shutters, acrylic or PU finishes run Rs 1.5-3 lakh, and "
            "stainless or quartz countertops add Rs 40,000-1,20,000. German hinges and "
            "soft-close channels are the highest-value upgrades; spend there before "
            "upgrading the shutter finish."
        ),
    },
    # ---- Room-specific ---------------------------------------------------
    {
        "category": "bedroom",
        "source": "bedroom",
        "content": (
            "Master bedroom design for Indian homes: an upholstered or wooden headboard "
            "wall in fluted paneling or cane, bedsheets in cotton or linen, and a layered "
            "bedside table rather than a bulky nightstand. Wardrobe should be floor-to-ceiling "
            "with sliding or openable shelving to maximise storage. A rug extending 60 cm "
            "beyond the bed on three sides anchors the room. Cost Rs 1.5-4 lakh mid-range."
        ),
    },
    {
        "category": "bedroom",
        "source": "bedroom",
        "content": (
            "Kids room design in India must be low-cost, non-toxic, and modular: a "
            "low-height bed (under 90 cm) for safety, rounded furniture corners, a "
            "dedicated reading corner with warm lighting, and storage in the form of "
            "under-bed drawers. Use washable cotton throws and a modular desk that can "
            "grow with the child. Budget Rs 50,000-1,50,000."
        ),
    },
    {
        "category": "home_office",
        "source": "home_office",
        "content": (
            "Home office design in India: a 4-5 ft desk, an ergonomic chair (avoid cheap "
            "unbranded ones for long WFH hours), a monitor arm, and cable management to "
            "avoid visual clutter. Natural light from a side window is preferred over a "
            "window directly behind the monitor. Add a pinboard or wall system for "
            "notes. Budget Rs 40,000-1,50,000."
        ),
    },
    {
        "category": "kitchen",
        "source": "kitchen",
        "content": (
            "Indian kitchen functional zones follow a 'work triangle' between sink, "
            "chopping board, and stove. A modular layout with tall units at the far end "
            "near the entry preserves the prep counter. Use granite or quartz countertops "
            "to withstand heat, and a deep sink (18 inch) for oversized vessels. "
            "Ventilation via a chimney or window near the stove is essential for oil-heavy "
            "cooking."
        ),
    },
    {
        "category": "living_room",
        "source": "living_room",
        "content": (
            "Indian living room convention: the TV unit wall is the focal point, with the "
            "sofa placed opposite. Use a low media console under 60 cm, a patterned rug "
            "with a 2-3 inch border, and sheer plus drape window treatments. Add brass or "
            "wood accents and indoor plants such as money plant, jade plant, or ZZ plant "
            "for low-maintenance greenery. Mix 2-3 textures per surface for visual depth."
        ),
    },
    {
        "category": "living_room",
        "source": "living_room",
        "content": (
            "Space-saving tricks for compact Indian living rooms: wall-mounted floating "
            "shelves instead of floor bookcases, a 2/3-seater sofa under 72 inches rather "
            "than a sectional, nesting or C-shaped coffee tables that tuck away, and "
            "curtains hung from ceiling to floor to make the window look taller. Vitrified "
            "tile flooring costs far less than marble and is easy to maintain. Keep a "
            "90 cm circulation path between the sofa and the TV unit."
        ),
    },
    {
        "category": "kids_room",
        "source": "kids_room",
        "content": (
            "Kids room safety and layout: use rounded furniture corners, keep the bed under "
            "90 cm high, wall-mount shelves above reach, and anchor tall storage units to "
            "the wall. Provide a dedicated reading nook with a bean bag or low armchair and "
            "a warm 3000K reading lamp. Choose washable cotton throws, and use under-bed "
            "drawers to add storage without extra floor area."
        ),
    },
    {
        "category": "home_office",
        "source": "home_office",
        "content": (
            "Ergonomics for a work-from-home desk: the monitor top should sit at or just "
            "below eye level about an arm's length away, the chair should support the lower "
            "back, and the feet should rest flat on the floor or a footrest. Manage cables "
            "with a desk-mounted tray and a cable sleeve to keep the floor clear for "
            "video calls. Layered lighting needs a task lamp on the desk plus ambient room "
            "light to avoid screen glare."
        ),
    },
    {
        "category": "kitchen",
        "source": "kitchen",
        "content": (
            "Ventilation is critical for heavy Indian cooking: install a chimney hood or "
            "chimney and glass draft shield directly above the stove, sized to at least "
            "the hob width. Use a granite or quartz countertop to withstand heat, a deep "
            "18-inch sink for large vessels, and a tall pull-out pantry unit so the prep "
            "counter stays clear. Modular shutters in laminate, acrylic, or PU with "
            "German soft-close hinges."
        ),
    },
    {
        "category": "storage",
        "source": "storage",
        "content": (
            "Indian homes need 25-30% of floor area as storage due to seasonal clothing and "
            "appliances. Solutions: loft storage above door frames, full-height wardrobes "
            "with a top shelf above 7 feet, modular under-stair units, and vertical "
            "shelving in balconies for dry goods. Kitchens need tall pull-out units for "
            "pantry items, a cutlery organiser, and a vertical tray rack."
        ),
    },
    {
        "category": "balcony",
        "source": "balcony",
        "content": (
            "Indian balconies handle monsoon drainage and heat. Use outdoor-rated "
            "furniture (teak or powder-coated metal), drainage-friendly planters, and "
            "mild-weatherproof finishes. Popular uses: a compact seating nook with "
            "string lights, a drying plus reading corner, or a vertical herb garden. Keep "
            "load under 200 kg per square metre for typical slabs."
        ),
    },
    {
        "category": "foyer",
        "source": "foyer",
        "content": (
            "Entrance foyer design in India should include a shoe cabinet, a console with a "
            "mirror for a final check, a key tray, and a cove-lit ceiling to feel welcoming. "
            "A vertical accent wall with slats or a mirror panel expands the narrow entry "
            "space common in Indian apartments. Add a small seating element or a tall "
            "indoor plant."
        ),
    },
    {
        "category": "sustainability",
        "source": "sustainability",
        "content": (
            "Sustainable and low-VOC options for Indian interiors: water-based low-VOC "
            "paints (Asian Paints Royale, Dulux Velvet Touch), bamboo or reclaimed wood "
            "furniture, jute rugs and cotton drapes, and modular flat-pack units designed "
            "for disassembly. Local craft in Kolhapuri, Jaipur blue pottery, or Kanjeevaram "
            "textiles supports artisans and adds regional character at moderate cost."
        ),
    },
    {
        "category": "accessibility",
        "source": "accessibility",
        "content": (
            "Ageing-in-place design for Indian apartments: keep the master bedroom and a "
            "bathroom on the same floor, add grab bars near the toilet and shower, use "
            "non-slip matte flooring, install lever handles instead of knobs, and ensure "
            "switch plates are at 900-1200 mm height. Replace thresholds with bevelled "
            "ramps under 13 mm to allow wheelchair or walker movement."
        ),
    },
    {
        "category": "principles",
        "source": "principles",
        "content": (
            "Core interior design principles: balance (visual weight distribution), "
            "proportion and scale (furniture sized to the room, roughly 2/3 the wall "
            "length for sofas), harmony (consistent 2-3 material palette), rhythm "
            "(repetition of colour, pattern, or form), emphasis (one focal point per "
            "room), and contrast. A successful room reads as intentional from the "
            "doorway."
        ),
    },
    {
        "category": "principles",
        "source": "principles",
        "content": (
            "Painting a room: always start from the highest point and work down. Cut in "
            "edges with a 2 inch angled brush, roll the middle section with a nap "
            "appropriate to the finish, and keep a wet edge to avoid lap marks. Indian "
            "climates need moisture-resistant primers on external walls. Two coats of "
            "emulsion over primer is the standard; putty must be sanded with 180-220 grit."
        ),
    },
]


def get_knowledge_base() -> List[Dict[str, str]]:
    """Return the raw knowledge base entries."""
    return KNOWLEDGE_BASE
