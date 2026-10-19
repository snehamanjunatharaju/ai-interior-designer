"""
Extended domain knowledge for Indian residential interiors.

Split from `knowledge_base.py` purely for file size; `get_knowledge_base()`
merges both. Every entry is written the same way as the original corpus -
specific, Indian, actionable, with measurements - because a passage that says
"choose high-quality materials" is worse than no passage at all: it consumes
a retrieval slot and teaches the model nothing.

Organisation follows the questions a designer is actually asked: how much
space does this need, what does it cost, what does Vastu say, what goes wrong,
and what does this room do for the people in it.
"""

from __future__ import annotations

from typing import Dict, List

EXTENDED_KNOWLEDGE: List[Dict[str, str]] = [
    # ==================================================================
    # Vastu - directions and placement
    # ==================================================================
    {
        "source": "Vastu, bedrooms",
        "category": "vastu",
        "content": (
            "Vastu prescribes the bed head in the south or south-west, so the "
            "sleeper faces north. The headboard must rest against a solid wall "
            "rather than a window or a gap. In a room where the south wall is "
            "taken by the door, the south-west corner is the second choice. "
            "Avoid placing the bed directly beneath a beam or a heavy "
            "wardrobe, which Vastu treats as oppressive overhead mass."
        ),
    },
    {
        "source": "Vastu, study and work",
        "category": "vastu",
        "content": (
            "A study desk belongs on an east or north wall, giving the person "
            "seated morning light and a clear forward view of the room. In "
            "Vastu the east is the direction of new beginnings and the north "
            "of wealth and intellect. A desk facing a wall removes distraction; "
            "a desk facing the door makes the worker feel observed. Place "
            "electronics on the desk's right and stationery on the left."
        ),
    },
    {
        "source": "Vastu, storage and heavy furniture",
        "category": "vastu",
        "content": (
            "Heavy storage - wardrobes, almirahs, safes - is placed in the "
            "south-west, the direction associated with stability. In Vastu "
            "nothing heavy should rest on the north-east, which is the most "
            "auspicious corner and is better kept light and open. Corner "
            "placement in the south-west also removes a dead zone in the room."
        ),
    },
    {
        "source": "Vastu, mirrors",
        "category": "vastu",
        "content": (
            "A mirror should not face a bed, which Vastu holds to reflect "
            "energy onto the sleeper at rest. Nor should it hang on the wall "
            "opposite a bed for the same reason. Mirrors on a side wall, or on "
            "the same wall as the bed head, are acceptable. A mirror placed low "
            "near the floor in a bedroom is avoided entirely, as is a mirror "
            "facing a toilet door."
        ),
    },
    {
        "source": "Vastu, entrance and door",
        "category": "vastu",
        "content": (
            "The main door is the most important opening in Vastu. A door on "
            "the north-east is most favourable; a south-west door is avoided. "
            "Nothing tall should stand in the door swing or immediately "
            "opposite the entrance, which obstructs the flow of energy. A "
            "small threshold mat, a wall-mounted nameplate, and a light just "
            "inside the entrance are traditional remedies. Vastu also advises "
            "a door closer so the door never swings wide into circulation."
        ),
    },
    {
        "source": "Vastu, kitchen",
        "category": "vastu",
        "content": (
            "The cooking platform faces south-east so the cook works with the "
            "sun to their left, which Vastu associates with fire and "
            "digestion. The sink or water inlet sits in the north-east. The "
            "refrigerator is placed in the north-west. Vastu separates the "
            "cooking platform from the entrance by a counter, and advises "
            "that the platform not be placed in the north-east corner. Gas "
            "should never sit directly in front of a window draught."
        ),
    },
    {
        "source": "Vastu, bathroom",
        "category": "vastu",
        "content": (
            "The toilet seat faces north or east, never south directly. The "
            "wash basin is placed in the north-east or east. A mirror sits on "
            "an east or north wall and never directly opposite the toilet. "
            "Vastu recommends a shower in the north-east corner of a bathroom "
            "and advises against a toilet in the south-west, the direction "
            "reserved for heavy storage in the home."
        ),
    },
    {
        "source": "Vastu, living room",
        "category": "vastu",
        "content": (
            "A living room TV unit is placed on the west or north wall so that "
            "seating faces east or south. The main sofa is positioned so the "
            "sitter faces north or east. Vastu keeps the north-east corner of "
            "a living room light - a plant or a floor lamp rather than a "
            "cupboard. Heavy seating occupies the south and west, leaving the "
            "north and east open for movement."
        ),
    },
    {
        "source": "Vastu, colour and finish",
        "category": "vastu",
        "content": (
            "Vastu associates colours with directions: white and cream in the "
            "north-east, green in the north, yellow and gold in the south-east, "
            "red and maroon in the south, blue and grey in the west. A heavy "
            "saffron or maroon wall is kept on the south or west wall. Vastu "
            "also favours finishes over patterns, and a single large artwork "
            "over many small pictures."
        ),
    },
    {
        "source": "Vastu, remedies and corrections",
        "category": "vastu",
        "content": (
            "Where a plan cannot follow the ideal direction, Vastu offers "
            "remedies rather than demanding demolition. A brass or copper "
            "object, a hanging lamp, a mirror, a bowl of water, or a specific "
            "colour on the relevant wall are traditional corrections. The "
            "practical sequence is: correct what can be corrected with layout, "
            "then with finish, then with a remedy. A consultant should not "
            "claim a remedy substitutes for a workable plan."
        ),
    },
    {
        "source": "Vastu, compass and measurement",
        "category": "vastu",
        "content": (
            "A Vastu reading needs the direction the main door faces, measured "
            "with a compass held flat at the threshold. Direction of a window "
            "is read as the compass bearing it looks towards, not the wall it "
            "sits on. Without the door and window bearings, and the room's "
            "length and width, a Vastu assessment is guesswork: rules for bed "
            "head, desk, storage, and mirror are all stated in terms of "
            "direction, so a plan without bearings cannot be scored honestly."
        ),
    },
    {
        "source": "Vastu, children's rooms",
        "category": "vastu",
        "content": (
            "A child's study desk faces east or north-east, and a child's bed "
            "head rests south or south-west. Vastu keeps a children's room "
            "light and uncluttered in the north-east corner. Desks are placed "
            "away from the bed so the child is not studying in the direct line "
            "of the sleeping position. Colours for a child's room should be "
            "bright and warm rather than the muted greys that read as austere."
        ),
    },

    # ==================================================================
    # Space planning and clearances
    # ==================================================================
    {
        "source": "Space planning, circulation",
        "category": "space_planning",
        "content": (
            "Maintain 90 cm of clear circulation on the main route through a "
            "room and 60 cm in front of secondary furniture. Below 60 cm a "
            "person must turn sideways, which is uncomfortable in a space used "
            "more than once a day. A bedroom needs 90 cm beside the bed on at "
            "least one side. Circulation should be checked with the door open, "
            "because a 90 cm leaf swings through a 90 cm arc."
        ),
    },
    {
        "source": "Space planning, door swing",
        "category": "space_planning",
        "content": (
            "A standard internal door leaf is 90 cm wide and swings through a "
            "quarter-circle of roughly 90 cm radius. Nothing taller than a "
            "shoe rack may stand inside that arc. A door opening into a "
            "bathroom should be an outward-opening or sliding door, since a "
            "swinging door needs floor space the bathroom rarely has. Sliding "
            "or pocket doors recover a full 1 square metre in a small bedroom."
        ),
    },
    {
        "source": "Space planning, small bedrooms",
        "category": "space_planning",
        "content": (
            "In a room under 100 square feet, a cot against the wall with "
            "storage beneath is the base unit - it keeps the floor clear while "
            "doubling as a bench. Wall-mounted shelving above 150 cm converts "
            "unusable air into storage. A mirror on the long wall visually "
            "lengthens the room. Avoid a wardrobe opposite the bed, which "
            "compresses the space between them to under 60 cm."
        ),
    },
    {
        "source": "Space planning, children's rooms",
        "category": "space_planning",
        "content": (
            "A child's room should keep 60 cm of clear floor between the bed "
            "and the desk for safe movement between the ages of four and "
            "twelve. Bunk beds must leave at least 90 cm of headroom above the "
            "upper berth, and a guard rail is mandatory above 90 cm. Storage "
            "below 90 cm should be anchored to the wall. A study desk at 72 cm "
            "suits ages six to twelve; above that, plan for 75 cm."
        ),
    },
    {
        "source": "Space planning, work from home",
        "category": "space_planning",
        "content": (
            "A workable home office needs 90 cm of depth for the desk plus 75 "
            "cm of chair clearance behind it - 165 cm total, or the chair hits "
            "the wall. The desk should not sit directly in front of a door or "
            "a window that opens inward. Background needs a wall behind the "
            "monitor; a desk facing a window causes glare. Cable routing is "
            "resolved in the wall, not taped to the skirting."
        ),
    },
    {
        "source": "Space planning, living rooms",
        "category": "space_planning",
        "content": (
            "In a living room under 150 square feet, a two-seater plus one "
            "armchair beats a three-seater, which crowds the walls. Anchor the "
            "seating around a rug rather than pushing everything to the "
            "perimeter - furniture floating in the middle of a small room "
            "reads as accidental. Leave 45 cm between a sofa and the coffee "
            "table, which is enough for a tray but not for walking."
        ),
    },
    {
        "source": "Space planning, kitchens",
        "category": "space_planning",
        "content": (
            "A single-platform Indian kitchen works at 3.6 m of counter run, "
            "giving roughly 1.2 m each for preparation, cooking, and washing. "
            "Between the counter edge and a wall there must be 100 cm to open "
            "an oven or a dishwasher door. The sink-to-hob distance should not "
            "exceed 2.1 m, or wet vegetables cross the cooking zone. Storage "
            "above 150 cm should be closed cupboards, not open shelves, near a "
            "hob, where grease settles."
        ),
    },

    # ==================================================================
    # Budget and value
    # ==================================================================
    {
        "source": "Budget, phasing purchases",
        "category": "budget",
        "content": (
            "Where a budget cannot cover the full wish list, the honest answer "
            "is a two-phase plan rather than a reduced scope presented as "
            "complete. Sleep comes first - a bed and a mattress - then the "
            "task light, then storage, then decor. Decor is the most "
            "deferrable spend and the least noticeable when absent. A client "
            "told 'this is not achievable at that budget' trusts the advice "
            "more than a proposal that fits on paper and fails in the store."
        ),
    },
    {
        "source": "Budget, where to save",
        "category": "budget",
        "content": (
            "On a constrained budget, spend on the bed, the mattress, the "
            "lighting, and the kitchen platform; save on decor, cushions, "
            "paint colour, and furniture finishes. Paint is the cheapest "
            "lever on perceived quality - a fresh coat in a considered colour "
            "costs a few thousand rupees and transforms a room. Avoid saving "
            "on electrical work, plumbing, or a study desk, where a poor "
            "purchase is used daily and is felt every day."
        ),
    },
    {
        "source": "Budget, realistic INR ranges",
        "category": "budget",
        "content": (
            "Typical 2026 Indian retail, excluding white goods: a single "
            "kids' cot 3,000-4,000 rupees, a single bed with storage "
            "7,000-9,500, a queen bed 17,000-22,000, a two-seater sofa "
            "17,000-24,000, a wardrobe 6,500-15,000 depending on width, a "
            "study desk 2,000-15,000, a modular kitchen base unit 18,000 per "
            "three feet. Interior wall paint is roughly 4,500 per room for two "
            "coats. Every figure should be confirmed at the retailer."
        ),
    },
    {
        "source": "Budget, renovation versus furniture",
        "category": "budget",
        "content": (
            "Civil work - plumbing, electrical rewiring, false ceiling, "
            "structural change - costs several times a full room refurnishing "
            "in an Indian apartment. A renter or a resident in a leased flat "
            "should spend on reversible furniture, peel-and-stick wall decals, "
            "and plug-and-play lighting, and skip anything requiring a "
            "contractor. False ceilings are the single most common cause of a "
            "renovation budget running out of control, and should be priced "
            "first and questioned hardest."
        ),
    },
    {
        "source": "Budget, hidden costs",
        "category": "budget",
        "content": (
            "Budgets routinely omit: electrical points and switches, socket "
            "additions, curtain rods and tracks, mattress and bedding, "
            "delivery and installation, assembly hardware, and the paint that "
            "must be redone after installation. A furnishing budget that "
            "allows 10-15 per cent for these is realistic; one that does not "
            "will appear to overrun on arrival."
        ),
    },
    {
        "source": "Budget, comparison shopping",
        "category": "budget",
        "content": (
            "The same wardrobe specification is commonly 30-40 per cent "
            "cheaper direct from a city factory outlet than through an "
            "franchise, with the difference in board grade and hardware "
            "warranty rather than visible finish. Compare like for like on "
            "carcass material, hinge make, and laminate thickness - a lower "
            "price on particleboard with a soft-close hinge is a real saving; "
            "a lower price on the same specification is simply a better deal."
        ),
    },

    # ==================================================================
    # Materials and finishes
    # ==================================================================
    {
        "source": "Materials, wood and boards",
        "category": "materials",
        "content": (
            "For Indian humidity, furniture carcass in BWP plywood or HDHMR "
            "outperiors MDF; particleboard swells at the edges and is only "
            "acceptable for closed, low units. Teak and sheesham are solid "
            "and stable but cost three to four times engineered board. "
            "Laminate is the budget choice and is serviceable on vertical "
            "surfaces; its weakness is chipping on exposed edges, which is "
            "addressed with edge banding. Veneer over MDF gives a solid-wood "
            "look at a mid price and can be re-polished."
        ),
    },
    {
        "source": "Materials, upholstery and fabric",
        "category": "materials",
        "content": (
            "Upholstery fabric in a home with children or pets should be a "
            "tightly woven cotton or a performance fabric with a stain "
            "guard, not linen or velvet, which mark and clean badly. Martindale "
            "of 30,000+ suits heavy use. Foam density matters more than the "
            "cover: 35-45 density in a seat is serviceable, below 30 compresses "
            "within a year. Removable covers are worth paying for where children "
            "are involved."
        ),
    },
    {
        "source": "Materials, tiles and flooring",
        "category": "materials",
        "content": (
            "Vitrified tiles suit kitchens and bathrooms; they are low "
            "porosity and need no sealing. Ceramic is cheaper but absorbs "
            "water. For bedrooms in an Indian climate, wooden or laminate "
            "flooring is warmer than tile, which feels cold underfoot in "
            "winter. Anti-skid rating matters in a bathroom - R10 or above. "
            "Wood flooring must be moisture-rated for the Indian monsoon, and "
            "an underlay is not optional."
        ),
    },
    {
        "source": "Materials, paint",
        "category": "materials",
        "content": (
            "Interior emulsion is washable, low-odour, and suited to Indian "
            "conditions; distemper is cheaper but not scrubbable. Two coats "
            "over a primer is the minimum that hides plaster unevenness. "
            "Matte finishes suit large rooms and hide joints; satin suits a "
            "room that is scrubbed often. High-traffic corridors take a "
            "washable sheen rather than flat matte, which shows every scuff."
        ),
    },

    # ==================================================================
    # Lighting
    # ==================================================================
    {
        "source": "Lighting, layers and colour temperature",
        "category": "lighting",
        "content": (
            "A usable room has three layers: ambient for general visibility, "
            "task for a specific activity, and accent for emphasis. Colour "
            "temperature matters more than brightness - 2700-3000K is warm and "
            "relaxing for living and bedrooms, 4000K neutral for kitchens and "
            "work, 6500K cool and clinical. Mixing temperatures in one room "
            "reads as a mistake. A single central ceiling light is the most "
            "common lighting failure in Indian homes."
        ),
    },
    {
        "source": "Lighting, children's study",
        "category": "lighting",
        "content": (
            "A child's study corner needs 400-500 lux at the desk, not the "
            "ambient level of the room. A 3000K task lamp with a CRI above 80 "
            "reduces eye strain for a child reading in the evening. The lamp "
            "sits on the opposite side to the writing hand to avoid a shadow "
            "across the page. Blinds or a sheer plus blackout are needed for "
            "an east window, which otherwise delivers direct low sun all "
            "morning."
        ),
    },
    {
        "source": "Lighting, windows and glare",
        "category": "lighting",
        "content": (
            "Light entering a window is directional and should be worked with: "
            "a north-east window gives soft even light all day, an east window "
            "low sun until noon, a west window harsh afternoon heat. A desk "
            "perpendicular to the window rather than facing it avoids glare on "
            "a screen. Curtains hung above and outside the frame make the "
            "window look larger and stop light leaking at the sides."
        ),
    },
    {
        "source": "Lighting, common faults",
        "category": "lighting",
        "content": (
            "The most frequent lighting faults in Indian homes: a single "
            "ceiling point per room with no task lighting; no switch at the "
            "bedhead, forcing a walk across a dark room; no socket where a "
            "lamp or a study light is needed; and downlights with no other "
            "layer, which flatten a space. Two-way switching on the bedroom "
            "door and bedhead is a small addition with a large effect."
        ),
    },

    # ==================================================================
    # Colour
    # ==================================================================
    {
        "source": "Colour, building a palette",
        "category": "color",
        "content": (
            "A workable interior palette is one neutral base, one secondary, "
            "and one accent - roughly 60-30-10 by area. The accent should be "
            "saturated enough to read as a colour rather than a tint, and used "
            "on less than a tenth of the surface. Neutrals are warm grey, "
            "greige, or off-white rather than pure white, which reads clinical "
            "under Indian daylight. Warming a palette with wood and brass "
            "keeps a cool grey from feeling unwelcoming."
        ),
    },
    {
        "source": "Colour, children's rooms",
        "category": "color",
        "content": (
            "A child's room should read as playful without becoming exhausting. "
            "One bright feature - an accent wall, a rug, a set of cushions - "
            "carries the character, while the walls stay a soft neutral the "
            "growth can outgrow. Coral, teal, and marigold work as a trio; so "
            "do mint and dusty pink. Avoid a saturated colour on all four "
            "walls, which leaves nothing for the furniture to add."
        ),
    },
    {
        "source": "Colour, testing and daylight",
        "category": "color",
        "content": (
            "Paint colours must be tested on the actual wall, at three times of "
            "day, because Indian daylight shifts a colour dramatically between "
            "a bright afternoon and a dim monsoon evening. A large sample of "
            "at least 30 cm square, painted in the same finish, is the minimum "
            "worth doing - a small dab is misleading. The same colour on a "
            "north-facing and a west-facing wall will not match."
        ),
    },
    {
        "source": "Colour, facing and finishes",
        "category": "color",
        "content": (
            "A west-facing room runs hot in the afternoon; cool colours - soft "
            "blue, sage, pale grey - reduce the sense of heat, and a light "
            "wall with blackout lining genuinely lowers the temperature. A "
            "north-facing room is even and cooler, and takes deeper colour "
            "without reading heavy. Satin and gloss both reflect more light; "
            "matte in a dark room absorbs it and reads gloomy."
        ),
    },

    # ==================================================================
    # Storage
    # ==================================================================
    {
        "source": "Storage, vertical space",
        "category": "storage",
        "content": (
            "In a room with a 270 cm ceiling, everything above 210 cm is "
            "storage that costs floor area nothing. Closed cabinets above "
            "200 cm hide seasonal clutter; open shelving is best between 120 "
            "and 200 cm, at a height reachable without a step ladder. The band "
            "below 90 cm is reserved for heavy items, and in a child's room it "
            "must be anchored. Storage between 90 and 120 cm is the "
            "least-usable band and is best left empty or shallow."
        ),
    },
    {
        "source": "Storage, wardrobes",
        "category": "storage",
        "content": (
            "A wardrobe depth of 55-60 cm takes a hanger on a rod; 60 cm is the "
            "practical minimum for long garments. Internal width of 90 cm "
            "allows a single hanging section; 120-150 cm allows a double "
            "hang. Loft storage above a 180 cm wardrobe gains a full cubic "
            "metre for seasonal bedding. Sliding doors save 60 cm of swing in "
            "a narrow bedroom; hinged doors need that space free."
        ),
    },
    {
        "source": "Storage, modular versus bespoke",
        "category": "storage",
        "content": (
            "Modular storage is adjustable and repairs one shelf at a time, "
            "but its internal width is fixed and its carcass quality varies. "
            "Bespoke carpentry fits an awkward room exactly and can use an "
            "awkward alcove that modular units cannot, at a higher upfront "
            "cost. For a standard rectangular room with normal dimensions, "
            "modular is better value; for an irregular room, bespoke recovers "
            "space that would otherwise be dead."
        ),
    },
    {
        "source": "Storage, modular kitchen",
        "category": "storage",
        "content": (
            "Modular kitchen storage is organised into zones: tall units for "
            "pantry at one end, base units for washing and preparation in the "
            "middle, and the cooking platform nearest the window. Drawer "
            "storage below the preparation zone is more usable than shelves at "
            "the same height - a drawer brings the item to you. A tall unit "
            "near the entrance is wasted circulation; against an end wall it "
            "is efficient."
        ),
    },

    # ==================================================================
    # Problems and failure modes
    # ==================================================================
    {
        "source": "Common problems, moisture",
        "category": "common_problems",
        "content": (
            "Condensation on windows in the monsoon is normal and is managed "
            "with ventilation, not with a dehumidifier permanently running. "
            "Paint bubbles near a window frame indicate a leak behind the "
            "plaster, not a paint problem. Skirting and the bottom 10 cm of "
            "wall are the first to show rising damp, and MDF skirting swells "
            "faster than PVC. In a ground-floor Indian home, a waterproof "
            "membrane behind skirting is worth the cost."
        ),
    },
    {
        "source": "Common problems, storage failures",
        "category": "common_problems",
        "content": (
            "The common storage failures in Indian homes: particleboard "
            "swelling at exposed edges, laminate peeling at the edges first, "
            "soft-close hinges failing because the door is out of square, "
            "shelves bowing when overloaded, and wall units falling because "
            "they were fixed into hollow gypsum rather than a stud. Specify "
            "BWP plywood interiors, stainless hinges, and a fixing point into "
            "a structural wall for anything above 5 kg."
        ),
    },
    {
        "source": "Common problems, electrical",
        "category": "common_problems",
        "content": (
            "Under-wired rooms show up as an extension board under the desk "
            "and a tripped MCB. An adult bedroom needs at least two socket "
            "points on each wall, a ceiling fan point at the centre, a bedhead "
            "socket, and a point for the air conditioner. A kitchen needs "
            "separate higher-capacity circuits for the oven, the microwave, "
            "and the kettle. Adding a socket after finishing is far more "
            "costly than planning for it, which is why it belongs in the "
            "renovation budget, not the furniture one."
        ),
    },
    {
        "source": "Common problems, space perception",
        "category": "common_problems",
        "content": (
            "A room reads small when it is over-furnished, when the furniture "
            "is all the same height, and when the ceiling is dark. Three "
            "corrections recover perceived space: furniture at two or three "
            "heights, a light ceiling with a slightly darker wall, and a large "
            "mirror on the wall opposite the window. Large-volume furniture "
            "such as a sectional pushes walls back perceptually, where an "
            "open-legged piece lets the eye travel past it."
        ),
    },
    {
        "source": "Common problems, neglect",
        "category": "common_problems",
        "content": (
            "Neglect, not design, is what makes a home look cheap. Mattresses "
            "are not rotated, curtains are not washed, sofa covers are not "
            "removed, grout is not scrubbed, and switchboards are never "
            "cleaned. A maintenance plan - quarterly mattress rotation, "
            "half-yearly deep clean, annual grout and switch cleaning - keeps "
            "an interior looking designed long after the photographs were "
            "taken."
        ),
    },
    {
        "source": "Common problems, rental apartments",
        "category": "common_problems",
        "content": (
            "In a rented Indian apartment, avoid anything requiring a "
            "contractor: no false ceiling, no wall breaking for plumbing, no "
            "painted feature wall. Work with adhesive vinyl for colour, "
            "freestanding shelf units for vertical storage, plug-in "
            "lighting, and washable throws rather than upholstered seating. "
            "Document the original paint colour and photograph switchboards at "
            "deposit, since a security deduction is otherwise hard to contest."
        ),
    },

    # ==================================================================
    # Indian market specifics
    # ==================================================================
    {
        "source": "Indian market, brands",
        "category": "market_india",
        "content": (
            "Established Indian furniture retail: Urban Ladder and Pepperfry "
            "for assembled mid-range and premium; Sleepcat and Nilkamal "
            "Homes for mattresses and storage; Hometown for compact urban "
            "furniture; Godrej Interio and Sleek for kitchens; Asian Paints "
            "and Berger for finishes; Jockey and Fabindia for textiles; Bajaj "
            "and Philips for lighting. Brand pricing for the same spec varies "
            "30-40 per cent, so compare carcass and hardware before finish."
        ),
    },
    {
        "source": "Indian market, apartment constraints",
        "category": "market_india",
        "content": (
            "An Indian apartment room is commonly 10x12 ft, 12x12 ft, or "
            "3.0x3.3 m, with a ceiling of 2.7 m and often a window on one wall "
            "only. Standard door width is 90 cm. Lift and stair dimensions "
            "limit which furniture can be delivered - a modular wardrobe over "
            "6 ft will often not fit in a lift, and flat-pack arrives in "
            "packets sized for a door. Check the delivery route before "
            "choosing a large assembled piece."
        ),
    },
    {
        "source": "Indian market, climate response",
        "category": "market_india",
        "content": (
            "Designing for Indian climate is mostly about sun and humidity. An "
            "east window needs a sheer and a blind; a west window needs "
            "blackout lining and a light-coloured exterior shade. Cross "
            "ventilation needs two openings on opposite walls, which in an "
            "apartment usually means the main door. Engineered wood at floor "
            "level needs a moisture-resistant finish. Ceiling fans at 270 mm "
            "with a 3-blade rotor move more air quietly than a decorative fan."
        ),
    },
    {
        "source": "Indian market, festival and pooja",
        "category": "market_india",
        "content": (
            "An Indian home needs somewhere for the puja, even a small one. A "
            "wall-mounted or floor-level unit in the north-east or east corner "
            "satisfies the Vastu placement, at 45-60 cm high and 30-45 cm "
            "wide. A study in a home office doubles as a puja shelf with a "
            "pull-out. Storage for festival lights, pooja samagri, and "
            "seasonal clothes is routinely the reason a home needs a second "
            "wardrobe, and it is forgotten until the week before Diwali."
        ),
    },
    {
        "source": "Indian market, family patterns",
        "category": "market_india",
        "content": (
            "Indian households are multi-generational more often than not, and "
            "a room designed only for the parents underuses its space. A "
            "child's room that will become a study should be planned with a "
            "desk height and a bed that a teenager will still accept. Ground "
            "floor rooms need a bed at 45-50 cm for an elderly parent, with "
            "space for a walking frame beside it. A home office that doubles "
            "as a guest room needs a sofa bed and a door that actually closes."
        ),
    },
    {
        "source": "Indian market, first homes",
        "category": "market_india",
        "content": (
            "A first home in India is often bought with a long view and "
            "finished on a short budget. The sequence that works: finish the "
            "kitchen and one bathroom properly, sleep on a good mattress, "
            "light the rooms properly, and furnish the rest as income allows. "
            "A living room used daily earns spend before a bedroom used on "
            "weekends. Buying a complete matching set at once is the common "
            "expensive mistake; buying one piece at a time in the same palette "
            "is not."
        ),
    },
    # ==================================================================
    # Kitchen ergonomics
    # ==================================================================
    {
        "source": "Kitchen ergonomics",
        "category": "kitchen",
        "content": (
            "The kitchen work triangle joins the sink, the hob and the "
            "refrigerator. Each leg should be 1.2-2.7 m and the sum under "
            "about 6.5 m, so a cook takes no more than two steps between the "
            "three. Counter height is 85-90 cm for adults and 80-85 cm for a "
            "short cook; the upper cabinets are hung 45-55 cm above the "
            "counter and no deeper than 35 cm, so the head is not at them."
        ),
    },
    {
        "source": "Kitchen ergonomics",
        "category": "kitchen",
        "content": (
            "A modular kitchen in an Indian apartment needs a minimum "
            "circulation of 90 cm between parallel counters, and 100 cm if "
            "two people cook together. Put daily-use items in a drawer bank "
            "between knee and eye level, not above the fridge. A tall unit "
            "holds a month of dry groceries; open shelves belong where jars "
            "are used daily, never over the hob where grease settles."
        ),
    },
    # ==================================================================
    # Ventilation for Indian cooking
    # ==================================================================
    {
        "source": "Kitchen ventilation, India",
        "category": "kitchen",
        "content": (
            "Indian cooking - tempering, deep frying, masala roasted in "
            "oil - produces far more grease and smoke than a Western hob, so "
            "a chimney is not optional in a semi-open kitchen. A 60 cm "
            "auto-clean chimney with 1,000-1,200 m3/hr suction is the minimum "
            "for a standard 90 cm hob; mount it 60-65 cm above the hob. "
            "Where ducting is impossible, a 250 mm exhaust fan through an "
            "external wall is the budget alternative and must not recirculate "
            "into the living room."
        ),
    },
    {
        "source": "Kitchen ventilation, India",
        "category": "kitchen",
        "content": (
            "Vastu places the kitchen in the south-east, called the agni "
            "corner, and asks the cook to face east at the hob. That working "
            "position also decides ventilation: the chimney run is short if "
            "the hob sits against the south-east external wall, and the "
            "window on the north-east supplies cross-ventilation. A hob on "
            "an internal wall needs a ducted run of 2-3 m and loses 20-30% "
            "of suction."
        ),
    },
    # ==================================================================
    # Japanese style
    # ==================================================================
    {
        "source": "Japanese interiors",
        "category": "style",
        "content": (
            "Japanese interiors are built on three ideas: restraint, low "
            "furniture and natural materials. A room reads Japanese when the "
            "floor is visible - low platform beds, a 30-35 cm high coffee "
            "table, seating that sits near the floor - and when storage is "
            "hidden behind sliding panels rather than shown as open shelving. "
            "Low furniture is not a style preference in a small apartment: "
            "it persists the sightline and makes a 9 x 11 ft room read larger."
        ),
    },
    {
        "source": "Japanese interiors",
        "category": "style",
        "content": (
            "The Japanese palette is quiet and comes from materials rather "
            "than paint: pale oak or walnut, rice paper, undyed cotton, "
            "charcoal and warm off-white, with a single accent such as "
            "matte black. Texture carries the room - a woven rug, a ceramic "
            "vase, linen curtains - because colour is deliberately limited. "
            "Wabi-sabi asks for one imperfect, handmade thing rather than a "
            "set of identical cushions."
        ),
    },
    # ==================================================================
    # Budget materials, Indian market
    # ==================================================================
    {
        "source": "Budget materials, India",
        "category": "materials",
        "content": (
            "Low-budget Indian interiors get their value from materials "
            "that survive daily use. Vitrified tiles give 60-70% of the look "
            "of marble at a fraction of the cost and do not stain. BWR "
            "plywood costs 20-30% more than commercial plywood and is the "
            "right choice for kitchen carcasses in a humid city. Laminate "
            "counters cost about a third of quartz and are adequate if seams "
            "are sealed; quartz is worth it around the sink."
        ),
    },
    {
        "source": "Budget materials, India",
        "category": "materials",
        "content": (
            "Where to save and where not to, in an Indian apartment on a "
            "tight budget: save on decor, wall art, lampshades and the "
            "sofa's upholstery fabric; spend on the mattress, the kitchen "
            "hardware (hinges and channels), the water-proofing and the "
            "wardrobe internals. The rule of thumb is that anything opened "
            "several times a day should be bought at the top of the budget, "
            "and anything looked at once a week can be bought cheap."
        ),
    },
    # ==================================================================
    # Kids' safety
    # ==================================================================
    {
        "source": "Kids' room safety",
        "category": "safety",
        "content": (
            "A children's room is designed against falls first. Anything "
            "taller than 60 cm - a bookshelf, a wardrobe, a chest of "
            "drawers - must be anchored to a studded wall; a five-drawer "
            "chest tip-over is caused by an open drawer, not by climbing. "
            "Beds are 40-45 cm high for a child under six, with a rail on "
            "the sleeping side, and the gap between the bed and the wall "
            "is closed so a child cannot wedge in."
        ),
    },
    {
        "source": "Kids' room safety",
        "category": "safety",
        "content": (
            "Choose materials a child can be around: low-VOC or zero-VOC "
            "paint with no added lead, water-based finishes on furniture, "
            "and no glass-topped tables. Rugs need an anti-skid backing on "
            "smooth tile. Socket covers and a blind cord cleat are cheap "
            "and are what parents actually notice; a bed lamp rated 3000K "
            "and set on a switch by the door lets a child move at night "
            "without a floor-level cable to trip on."
        ),
    },
]
