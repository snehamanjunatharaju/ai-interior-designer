"""
Seed catalog data.

IMPORTANT - READ BEFORE DEMOING
------------------------------
These prices are *indicative reference figures* compiled from publicly listed
Indian retail pricing, rounded to the nearest ₹50. They are NOT scraped from a
live feed and they are not guaranteed current. Before showing this to a client
or an interviewer as "real prices", verify a sample against the live sites.

What matters for correctness of the *system* is unaffected: every price in the
catalog is a stored integer that Python sums, never a number a language model
produced. Swapping this file for a real feed later changes values, not
behaviour.

Dimensions are in centimetres and reflect typical assembled sizes, which is
what the layout engine needs. Where a product is sold flat-packed, the note
records the packed size rather than the assembled one.
"""

from __future__ import annotations

from app.catalog.store import (
    TIER_ESSENTIAL,
    TIER_STANDARD,
    TIER_UPGRADE,
    CatalogItem,
)

# Room types each catalog category belongs to.
#
# Categories not listed here are universal (a lamp, a rug, paint). The list
# exists because a kitchen brief used to be able to buy a kids' study table
# and a bean bag: the model was shown a catalog with no notion of *where* a
# product belongs, so nothing stopped it. Filtering is applied on the way into
# the prompt (`room_catalog`) and again by the validator.
CATEGORY_ROOMS: dict[str, tuple[str, ...]] = {
    "bed": ("bedroom", "kids_room"),
    "mattress": ("bedroom", "kids_room"),
    "desk": ("kids_room", "home_office", "bedroom"),
    "chair": ("kids_room", "home_office", "dining", "bedroom"),
    "wardrobe": ("bedroom", "kids_room"),
    "tv_unit": ("living_room", "bedroom"),
    "storage": (
        "bedroom", "kids_room", "home_office", "living_room", "kitchen", "entrance",
    ),
    "seating": ("living_room", "balcony", "bedroom", "kids_room"),
    "table": ("living_room", "dining", "bedroom", "home_office"),
    "kitchen": ("kitchen", "dining"),
}

# Who each category is for, independent of room. A bean bag or a coffee table
# is furniture for a living room, not for a kitchen, whatever the audience.
CATEGORY_AUDIENCE: dict[str, str] = {}

# Everyday names -> SKU. Used by price lookup and by the refine step, so
# "chimney" and "countertop" resolve to the catalogue product a client means.
ITEM_ALIASES: dict[str, str] = {}

SEED_ITEMS: list[CatalogItem] = [
    # ======================================================================
    # Beds - kids and single
    # ======================================================================
    CatalogItem(
        "BED-KID-COT-01", "Kids Single Cot 90cm", "bed", "Sleepwell",
        3200, 90, 190, 60, TIER_ESSENTIAL,
        "https://www.sleepwell.com/",
        "Low bed height suits children; safety rail recommended", audience="kids"
    ),
    CatalogItem(
        "BED-KID-BUNK-01", "Kids Bunk Bed (upper + lower)", "bed", "Sleepwell",
        14500, 100, 200, 160, TIER_UPGRADE,
        "https://www.sleepwell.com/",
        "Frees floor space; ladder on the wall side", 2, audience="kids"
    ),
    CatalogItem(
        "BED-SGL-PLY-01", "Single Bed - Plywood with Storage", "bed", "Urban Ladder",
        7800, 90, 200, 40, TIER_ESSENTIAL,
        "https://urbanladder.com/",
        "Hydraulic storage lift; slides on castors",
    ),
    CatalogItem(
        "BED-SGL-SHEET-01", "Single Bed - Sheesham Finish", "bed", "Nilkamal",
        9500, 90, 200, 40, TIER_STANDARD,
        "https://www.nilkamalfurniture.com/",
    ),
    CatalogItem(
        "BED-DBL-QUEEN-01", "Queen Bed - Walnut Finish", "bed", "Sleepcat",
        18500, 150, 210, 40, TIER_STANDARD,
        "https://www.sleepcat.in/",
    ),
    CatalogItem(
        "MAT-SGL-01", "Single Mattress - 4in Medium Firm", "mattress", "Sleepcat",
        4500, 90, 190, 12, TIER_ESSENTIAL,
        "https://www.sleepcat.in/",
    ),
    CatalogItem(
        "MAT-KID-01", "Kids Mattress 2in - Comfort", "mattress", "Kurlon",
        2500, 80, 160, 8, TIER_ESSENTIAL,
        "https://www.kurlon.com/", audience="kids"
    ),

    # ======================================================================
    # Study / desk
    # ======================================================================
    CatalogItem(
        "DSK-KID-01", "Kids Study Table 90cm", "desk", "Hometown",
        2100, 90, 55, 72, TIER_ESSENTIAL,
        "https://www.hometownfurniture.com/",
        "Height 72cm suits ages 6-12; pairs with a 32cm chair", audience="kids"
    ),
    CatalogItem(
        "DSK-STD-01", "Study Desk 120cm with Drawer", "desk", "Urban Ladder",
        6900, 120, 60, 75, TIER_STANDARD,
        "https://urbanladder.com/",
    ),
    CatalogItem(
        "DSK-ERG-01", "Ergonomic Study Desk (height adjustable)", "desk", "Green Laundry",
        14500, 120, 60, 75, TIER_UPGRADE,
        "https://greenlaundry.in/",
    ),
    CatalogItem(
        "CHR-KID-01", "Kids Chair - Ergonomic", "chair", "Hometown",
        1400, 38, 45, 60, TIER_ESSENTIAL,
        "https://www.hometownfurniture.com/", audience="kids"
    ),
    CatalogItem(
        "CHR-OFF-01", "Office Chair - Mesh Back", "chair", "Green Laundry",
        4800, 55, 55, 110, TIER_STANDARD,
        "https://greenlaundry.in/",
    ),
    CatalogItem(
        "LMP-DESK-01", "Desk Lamp - 3000K Warm White", "lighting", "Bajaj",
        900, 15, 15, 40, TIER_ESSENTIAL,
        "https://www.bajaj.com/",
        "3000K recommended for a child's study corner",
    ),
    CatalogItem(
        "LMP-NIGHT-01", "Night Lamp - Bedside", "lighting", "Bajaj",
        700, 15, 15, 30, TIER_ESSENTIAL,
        "https://www.bajaj.com/",
    ),

    # ======================================================================
    # Storage
    # ======================================================================
    CatalogItem(
        "WRD-2D-01", "Wardrobe 2-Door - 3ft", "wardrobe", "Nilkamal",
        6500, 90, 55, 180, TIER_UPGRADE,
        "https://www.nilkamalfurniture.com/",
    ),
    CatalogItem(
        "WRD-2D-02", "Wardrobe 2-Door - 4ft", "wardrobe", "Sleepcat",
        11500, 120, 58, 200, TIER_UPGRADE,
        "https://www.sleepcat.in/",
    ),
    CatalogItem(
        "SHV-BOOK-01", "Bookshelf 3-Tier - 60cm", "storage", "Urban Ladder",
        2400, 60, 30, 90, TIER_UPGRADE,
        "https://urbanladder.com/",
        "Must be anchored to a studded wall if children use it",
    ),
    CatalogItem(
        "CHV-STO-01", "Toy Storage Chest - 70cm", "storage", "Hometown",
        1800, 70, 40, 45, TIER_ESSENTIAL,
        "https://www.hometownfurniture.com/", audience="kids"
    ),
    CatalogItem(
        "STO-CHV-01", "Cube Storage - 5 Cubes", "storage", "Urban Ladder",
        2900, 75, 30, 75, TIER_STANDARD,
        "https://urbanladder.com/",
    ),
    CatalogItem(
        "STO-SHLF-01", "Floating Wall Shelf 90cm", "storage", "Hometown",
        1300, 90, 20, 25, TIER_ESSENTIAL,
        "https://www.hometownfurniture.com/",
    ),
    CatalogItem(
        "SHR-ITSY-3T-01", "Nilkamal Itsy 3-Tier Shoe Rack", "storage", "Nilkamal",
        1700, 50, 28, 61, TIER_ESSENTIAL,
        "https://www.nilkamalfurniture.com/products/nilkamal-itsy-3-tier-shoe-rack",
        "Indicative listed price checked 2026-10-20; dimensions are typical assembled estimate",
    ),

    # ======================================================================
    # Seating
    # ======================================================================
    CatalogItem(
        "SOF-2S-01", "2-Seater Sofa - Fabric", "seating", "Urban Ladder",
        18500, 150, 85, 80, TIER_UPGRADE,
        "https://urbanladder.com/",
    ),
    CatalogItem(
        "ARM-01", "Armchair - Single", "seating", "Sleepcat",
        9500, 75, 80, 80, TIER_UPGRADE,
        "https://www.sleepcat.in/",
    ),
    # Domestic bean bags are sold by upholstery/ready-made brands, not by
    # furnishing retailers - Fabindia selling one was exactly the kind of
    # invented brand association that made the list untrustworthy.
    CatalogItem(
        "BEAN-01", "Bean Bag - Large", "seating", "Wakefit",
        3500, 90, 90, 70, TIER_ESSENTIAL,
        "https://www.wakefit.co/",
    ),
    CatalogItem(
        "STL-01", "Stool - Wood, 45cm", "seating", "Urban Ladder",
        1600, 40, 40, 45, TIER_ESSENTIAL,
        "https://urbanladder.com/",
    ),
    CatalogItem(
        "POUF-01", "Pouf - Handwoven", "seating", "Fabindia",
        2200, 45, 45, 35, TIER_ESSENTIAL,
        "https://www.fabindia.com/",
    ),

    # ======================================================================
    # Tables
    # ======================================================================
    CatalogItem(
        "TBL-COF-01", "Coffee Table - Sheesham", "table", "Urban Ladder",
        6800, 100, 50, 40, TIER_UPGRADE,
        "https://urbanladder.com/",
    ),
    CatalogItem(
        "TBL-SID-01", "Side Table - Round 40cm", "table", "Hometown",
        2400, 40, 40, 50, TIER_ESSENTIAL,
        "https://www.hometownfurniture.com/",
    ),
    CatalogItem(
        "DIN-TBL-01", "Dining Table 4-Seater", "table", "Nilkamal",
        14500, 120, 80, 75, TIER_UPGRADE,
        "https://www.nilkamalfurniture.com/",
    ),

    # ======================================================================
    # TV / electronics
    # ======================================================================
    CatalogItem(
        "TVU-WAL-01", "TV Unit - Wall Mounted 4ft", "tv_unit", "Urban Ladder",
        3200, 120, 35, 45, TIER_UPGRADE,
        "https://urbanladder.com/",
        "Wall-mounted; needs a stud or anchor into a solid wall",
    ),
    CatalogItem(
        "TVU-STD-01", "TV Stand - Low, 3ft", "tv_unit", "Nilkamal",
        4600, 90, 40, 45, TIER_UPGRADE,
        "https://www.nilkamalfurniture.com/",
    ),

    # ======================================================================
    # Soft furnishings
    # ======================================================================
    CatalogItem(
        "RUG-KID-01", "Kids Round Rug 150cm", "soft", "Hometown",
        1400, 150, 150, 2, TIER_ESSENTIAL,
        "https://www.hometownfurniture.com/",
        "Anti-skid backing recommended on smooth tile", audience="kids"
    ),
    CatalogItem(
        "RUG-LIV-01", "Living Room Rug 240x170cm", "soft", "Urban Ladder",
        4200, 240, 170, 2, TIER_UPGRADE,
        "https://urbanladder.com/",
    ),
    CatalogItem(
        "CUR-KID-01", "Curtains - Blackout, Per Pair", "soft", "Jockey",
        1300, 140, 2, 220, TIER_ESSENTIAL,
        "https://www.jockey.com/",
        "Blackout for a child's room controls the 7am wake-up problem", audience="kids"
    ),
    CatalogItem(
        "QUIL-KID-01", "Kids Quilt - Cotton", "soft", "Fabindia",
        1100, 100, 150, 5, TIER_ESSENTIAL,
        "https://www.fabindia.com/", audience="kids"
    ),

    # ======================================================================
    # Decor
    # ======================================================================
    CatalogItem(
        "DEC-FRM-01", "Framed Wall Art 60x40cm", "decor", "Decdn",
        1200, 60, 3, 40, TIER_ESSENTIAL,
        "https://www.decdn.in/",
    ),
    CatalogItem(
        "DEC-PLT-01", "Planter with Indoor Plant", "decor", "Ugao",
        650, 20, 20, 35, TIER_ESSENTIAL,
        "https://www.ugao.com/",
    ),
    CatalogItem(
        "DEC-MIR-01", "Wall Mirror 60x80cm", "decor", "Urban Ladder",
        1900, 60, 3, 80, TIER_ESSENTIAL,
        "https://urbanladder.com/",
        "Vastu: never place directly opposite a bed head",
    ),
    CatalogItem(
        "DEC-DSL-01", "Wall Decal Set - Kids", "decor", "Wallsta",
        600, 120, 1, 100, TIER_ESSENTIAL,
        "https://www.wallsta.com/",
        "Removable adhesive vinyl; renter friendly", audience="kids"
    ),
    CatalogItem(
        "DEC-BIN-01", "Laundry Hamper - Bamboo", "decor", "Urban Ladder",
        1300, 40, 40, 60, TIER_ESSENTIAL,
        "https://urbanladder.com/",
    ),

    # ======================================================================
    # Lighting
    # ======================================================================
    CatalogItem(
        "LMP-CEL-01", "Ceiling Light - Warm White LED", "lighting", "Bajaj",
        1800, 30, 30, 8, TIER_ESSENTIAL,
        "https://www.bajaj.com/",
    ),
    CatalogItem(
        "LMP-FLO-01", "Floor Lamp - Tripod", "lighting", "Bajaj",
        3400, 40, 40, 150, TIER_UPGRADE,
        "https://www.bajaj.com/",
    ),
    CatalogItem(
        "LMP-STR-01", "String Lights 5m", "lighting", "Bajaj",
        450, 500, 2, 2, TIER_ESSENTIAL,
        "https://www.bajaj.com/",
    ),

    # ======================================================================
    # Paint / renovation
    # ======================================================================
    CatalogItem(
        "PNT-INT-01", "Interior Wall Paint - Per Room", "paint", "Asian Paints",
        4500, 300, 300, 270, TIER_ESSENTIAL,
        "https://www.asianpaints.com/",
        "Covers one room up to 130 sq ft, 2 coats",
    ),
    CatalogItem(
        "PNT-PTL-01", "Ceiling Paint - Per Room", "paint", "Asian Paints",
        2200, 300, 300, 270, TIER_ESSENTIAL,
        "https://www.asianpaints.com/",
    ),

    # ======================================================================
    # Modular kitchen (kept small - a kids room brief should not match these)
    # ======================================================================
    # Brands are the ones that actually sell each product. Hettich makes
    # hinges and carcass hardware, not countertops; a countertop brand is a
    # stone/premium-surface supplier such as KalingaStone, and a sink is a
    # plumbing brand such as Jaquar.
    CatalogItem(
        "KIT-BSE-01", "Modular Kitchen Base Unit 3ft", "kitchen", "Hettich",
        18000, 90, 60, 85, TIER_UPGRADE,
        "https://www.hettich.com/",
        "Carcass plus soft-close hardware; corner unit available",
    ),
    CatalogItem(
        "KIT-CNT-01", "Kitchen Countertop - Per 3ft", "kitchen", "KalingaStone",
        9500, 90, 60, 4, TIER_UPGRADE,
        "https://www.kalingastone.com/",
        "Quartz slab; cut-out for sink and hob priced separately",
    ),
    CatalogItem(
        "KIT-SNK-01", "Kitchen Sink with Tap", "kitchen", "Jaquar",
        6500, 60, 45, 20, TIER_UPGRADE,
        "https://www.jaquar.com/",
    ),
    CatalogItem(
        "KIT-OVC-01", "Overhead Wall Cabinet 3ft", "kitchen", "Godrej Interio",
        12500, 90, 35, 70, TIER_UPGRADE,
        "https://www.godrejinterio.com/",
        "Wall-hung; set 45-55cm above the countertop",
    ),
    CatalogItem(
        "KIT-SHLF-01", "Open Kitchen Shelf 3ft", "kitchen", "Nilkamal",
        2200, 90, 25, 4, TIER_ESSENTIAL,
        "https://www.nilkamalfurniture.com/",
        "For daily-use jars and crockery; keep below eye level",
    ),
    CatalogItem(
        "KIT-PNT-01", "Tall Pantry Unit", "kitchen", "Godrej Interio",
        16500, 60, 60, 210, TIER_UPGRADE,
        "https://www.godrejinterio.com/",
        "Pull-out baskets; holds a month of dry groceries",
    ),
    CatalogItem(
        "KIT-TRL-01", "Kitchen Trolley - 3 Tier", "kitchen", "Nilkamal",
        1900, 45, 40, 80, TIER_ESSENTIAL,
        "https://www.nilkamalfurniture.com/",
        "Wheels; doubles as a serving cart",
    ),
    CatalogItem(
        "KIT-EXH-01", "Auto-Clean Chimney 60cm", "kitchen", "Elica",
        13500, 60, 45, 60, TIER_UPGRADE,
        "https://www.elica.com/",
        "Essential for Indian cooking; mount 65cm above the hob",
    ),
    CatalogItem(
        "KIT-FAN-01", "Kitchen Exhaust Fan 250mm", "kitchen", "Bajaj",
        2200, 30, 30, 30, TIER_ESSENTIAL,
        "https://www.bajaj.com/",
        "Budget alternative to a chimney; needs a wall opening",
    ),
    CatalogItem(
        "KIT-BSP-01", "Kitchen Backsplash - Per 3ft", "kitchen", "KalingaStone",
        3200, 90, 2, 60, TIER_STANDARD,
        "https://www.kalingastone.com/",
        "Wipe-clean surface behind the hob; up to 60cm tall",
    ),
    CatalogItem(
        "KIT-LGT-01", "Under-Cabinet Task Light 60cm", "kitchen", "Wipro",
        1100, 60, 5, 2, TIER_ESSENTIAL,
        "https://www.wiproconsumerlighting.com/",
        "4000K LED; lights the countertop, not the ceiling",
    ),
]

# Everyday names a brief, the model, or a refine request uses.
ITEM_ALIASES.update({
    "chimney": "KIT-EXH-01",
    "kitchen chimney": "KIT-EXH-01",
    "exhaust": "KIT-FAN-01",
    "exhaust fan": "KIT-FAN-01",
    "overhead cabinet": "KIT-OVC-01",
    "overhead cabinets": "KIT-OVC-01",
    "wall cabinet": "KIT-OVC-01",
    "open shelf": "KIT-SHLF-01",
    "open shelves": "KIT-SHLF-01",
    "pantry": "KIT-PNT-01",
    "pantry unit": "KIT-PNT-01",
    "trolley": "KIT-TRL-01",
    "kitchen trolley": "KIT-TRL-01",
    "backsplash": "KIT-BSP-01",
    "back splash": "KIT-BSP-01",
    "countertop": "KIT-CNT-01",
    "granite counter": "KIT-CNT-01",
    "sink": "KIT-SNK-01",
    "base unit": "KIT-BSE-01",
    "bean bag": "BEAN-01",
    "beanbag": "BEAN-01",
})
