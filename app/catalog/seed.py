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

SEED_ITEMS: list[CatalogItem] = [
    # ======================================================================
    # Beds - kids and single
    # ======================================================================
    CatalogItem(
        "BED-KID-COT-01", "Kids Single Cot 90cm", "bed", "Sleepwell",
        3200, 90, 190, 60, TIER_ESSENTIAL,
        "https://www.sleepwell.com/",
        "Low bed height suits children; safety rail recommended",
    ),
    CatalogItem(
        "BED-KID-BUNK-01", "Kids Bunk Bed (upper + lower)", "bed", "Sleepwell",
        14500, 100, 200, 160, TIER_UPGRADE,
        "https://www.sleepwell.com/",
        "Frees floor space; ladder on the wall side", 2,
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
        "https://www.kurlon.com/",
    ),

    # ======================================================================
    # Study / desk
    # ======================================================================
    CatalogItem(
        "DSK-KID-01", "Kids Study Table 90cm", "desk", "Hometown",
        2100, 90, 55, 72, TIER_ESSENTIAL,
        "https://www.hometownfurniture.com/",
        "Height 72cm suits ages 6-12; pairs with a 32cm chair",
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
        "https://www.hometownfurniture.com/",
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
        "https://www.hometownfurniture.com/",
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
    CatalogItem(
        "BEAN-01", "Bean Bag - Large", "seating", "Fabindia",
        3500, 90, 90, 70, TIER_ESSENTIAL,
        "https://www.fabindia.com/",
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
        "Anti-skid backing recommended on smooth tile",
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
        "Blackout for a child's room controls the 7am wake-up problem",
    ),
    CatalogItem(
        "QUIL-KID-01", "Kids Quilt - Cotton", "soft", "Fabindia",
        1100, 100, 150, 5, TIER_ESSENTIAL,
        "https://www.fabindia.com/",
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
        "Removable adhesive vinyl; renter friendly",
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
    CatalogItem(
        "KIT-BSE-01", "Modular Kitchen Base Unit 3ft", "kitchen", "Hettich",
        18000, 90, 60, 85, TIER_UPGRADE,
        "https://www.hettich.com/",
    ),
    CatalogItem(
        "KIT-CNT-01", "Kitchen Countertop - Per 3ft", "kitchen", "Hettich",
        9500, 90, 60, 4, TIER_UPGRADE,
        "https://www.hettich.com/",
    ),
    CatalogItem(
        "KIT-SNK-01", "Kitchen Sink with Tap", "kitchen", "Hettich",
        6500, 60, 45, 20, TIER_UPGRADE,
        "https://www.hettich.com/",
    ),
]
