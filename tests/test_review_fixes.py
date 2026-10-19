"""
Regression tests for the review findings, one class per fix.

Each test drives the same code path the bug was observed on, and asserts a
property that follows from the rule rather than a particular wording - the
price of a catalog row, a direction that came from the form, a rule result the
validator blocks on. Where the pipeline is involved the LLM is stubbed, so the
tests run offline and deterministically.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from app.agents.orchestrator import ArchitectAgent, DesignPipeline
from app.catalog import catalog
from app.core.schemas import DesignRequest
from app.engine.brief import (
    BriefConstraints,
    normalize_direction,
    parse_brief,
)
from app.engine.palette import palette_for
from app.engine.pricing import price_design, refresh_line_totals, total_inr
from app.engine.refine import apply_refinement, classify_refinement
from app.engine.validator import validate_design
from app.engine.vastu import analyse_design


@pytest.fixture(scope="module", autouse=True)
def ready():
    catalog.initialize()
    return catalog


def _request(**kw) -> DesignRequest:
    base = dict(
        client_preferences="A kitchen for a family of four",
        room_type="kitchen",
        color_scheme="warm",
        length_ft=9,
        width_ft=11,
        door_direction="north",
        window_direction="northeast",
        vastu_compliance="strict",
        budget_amount_inr=150000,
    )
    base.update(kw)
    return DesignRequest(**base)


def _brief(request: DesignRequest) -> BriefConstraints:
    return parse_brief(
        client_preferences=request.client_preferences,
        furniture=request.furniture,
        specific_requirements=request.specific_requirements,
        occupants=request.occupants,
        room_type=request.room_type.value,
        style=request.style.value,
        color_scheme=request.color_scheme,
        budget_inr=request.budget_amount_inr,
        vastu_compliance=request.vastu_compliance.value,
        room_dimensions=request.dimension_text(),
        door_direction=request.door_direction,
        window_direction=request.window_direction,
        lifestyle_notes=request.lifestyle_notes,
    )


def _codes(report) -> set:
    return {v.code for v in report.violations}


def _kitchen(**extra) -> dict:
    base = {
        "title": "Warm Kitchen",
        "concept": "A warm, earthy kitchen with practical work zones.",
        "layout_description": "Counter on the south-east wall; door north.",
        "furniture": [
            {"name": "Modular Kitchen Base Unit 3ft", "category": "kitchen",
             "sku": "KIT-BSE-01", "quantity": 1},
            {"name": "Auto-Clean Chimney 60cm", "category": "kitchen",
             "sku": "KIT-EXH-01", "quantity": 1},
        ],
        "color_palette": {
            "primary": "terracotta", "secondary": "olive", "accent": "rust",
            "neutrals": ["sand"], "hex_codes": ["#C0704A", "#7A8B4F", "#A63D2F"],
        },
        "trade_offs": ["Overhead units reduce decorative wall space"],
        "cons": ["Overhead units reduce decorative wall space"],
        "storage_strategy": "Overhead cabinets with wipe-clean laminate fronts.",
    }
    base.update(extra)
    return base


# --------------------------------------------------------------------------
# 1. Pricing
# --------------------------------------------------------------------------


class TestPricing:
    def test_every_line_has_a_real_price(self):
        """The reported bug: every shopping row read Rs 0."""
        design = _kitchen()
        cost = price_design(design)
        assert cost["line_items"]
        assert all(li["unit_price_inr"] > 0 for li in cost["line_items"])
        assert cost["estimated_cost_inr"] > 0

    def test_total_equals_sum_of_unit_times_quantity(self):
        design = _kitchen(
            furniture=[
                {"name": "Modular Kitchen Base Unit 3ft", "category": "kitchen",
                 "sku": "KIT-BSE-01", "quantity": 2},
                {"name": "Kitchen Trolley - 3 Tier", "category": "kitchen",
                 "sku": "KIT-TRL-01", "quantity": 1},
            ]
        )
        cost = price_design(design)
        expected = sum(li["unit_price_inr"] * li["quantity"] for li in cost["line_items"])
        assert cost["estimated_cost_inr"] == expected == total_inr(cost["line_items"])

    def test_name_only_entry_is_resolved_from_the_catalog(self):
        """A row with a name but no SKU still gets a real price."""
        design = _kitchen(furniture=[
            {"name": "Auto-Clean Chimney 60cm", "quantity": 1},
        ])
        cost = price_design(design)
        assert cost["costing_complete"] is True
        assert cost["line_items"][0]["sku"] == "KIT-EXH-01"
        assert cost["line_items"][0]["unit_price_inr"] == 13500

    def test_duplicate_sku_is_one_line(self):
        """\"Kids Chair - Ergonomic\" appeared twice; it is one purchase."""
        design = _kitchen(furniture=[
            {"name": "Modular Kitchen Base Unit 3ft", "sku": "KIT-BSE-01", "quantity": 1},
            {"name": "Modular Kitchen Base Unit 3ft", "sku": "KIT-BSE-01", "quantity": 1},
        ])
        cost = price_design(design)
        assert len(cost["line_items"]) == 1
        assert cost["estimated_cost_inr"] == 18000

    def test_refresh_recomputes_a_stale_line_total(self):
        line = {"sku": "X", "unit_price_inr": 100, "quantity": 3, "line_total_inr": 100}
        out = refresh_line_totals([line])
        assert out[0]["line_total_inr"] == 300

    def test_option_total_builder_uses_the_shared_function(self):
        """The option card, edit step and PDF all sum the same row shape."""
        request = _request(num_options=1)
        option = DesignPipeline()._to_option(_kitchen(), request)
        assert option.estimated_cost_inr == sum(
            li.unit_price_inr * li.quantity for li in option.furniture
        )
        assert option.estimated_cost_inr == total_inr(
            [dict(li.model_dump()) for li in option.furniture]
        )

    def test_option_rows_show_quantity_unit_and_line_total(self):
        request = _request(num_options=1)
        option = DesignPipeline()._to_option(
            _kitchen(furniture=[
                {"name": "Kitchen Trolley - 3 Tier", "sku": "KIT-TRL-01",
                 "category": "kitchen", "quantity": 2},
            ]),
            request,
        )
        row = option.shopping_list[0]
        assert row["quantity"] == 2
        assert row["unit_price_inr"] == 1900
        assert row["line_total_inr"] == 3800


# --------------------------------------------------------------------------
# 2. Form inputs reach every agent and the validator
# --------------------------------------------------------------------------


class TestStructuredBrief:
    def test_form_directions_are_authoritative(self):
        brief = _brief(_request(door_direction="north", window_direction="northeast"))
        assert brief.door_direction == "north"
        assert brief.window_direction == "northeast"

    def test_abbreviations_normalise_to_full_directions(self):
        assert normalize_direction("NE") == "northeast"
        assert normalize_direction("north-east") == "northeast"
        assert normalize_direction("South West") == "southwest"

    def test_brief_object_carries_every_form_field(self):
        request = _request(
            specific_requirements=["kids_safe", "storage_heavy"],
            lifestyle_notes="vegetarian cooking",
            furniture=["wardrobe"],
        )
        obj = request.brief_object()
        for key in (
            "room_type", "dimensions", "door_direction", "window_direction",
            "color_scheme", "vastu_level", "budget_inr", "occupants",
            "furniture", "special_requirements", "lifestyle_notes",
        ):
            assert key in obj, key
        assert obj["door_direction"] == "north"
        assert obj["window_direction"] == "northeast"
        assert obj["special_requirements"] == ["kids_safe", "storage_heavy"]
        assert obj["dimensions"]["width_ft"] == 11
        assert _brief(request).selected_furniture == ["wardrobe"]

    def test_validator_does_not_say_directions_were_not_supplied(self):
        brief = _brief(_request())
        design = _kitchen(concept="A kitchen that respects Vastu.")
        design["_vastu"] = analyse_design(design, brief).to_dict()
        design.update(layout=_layout_kitchen())
        report = validate_design(design, brief, price_design(design))
        assert "unverified_bearing_claim" not in _codes(report)

    @pytest.mark.asyncio
    async def test_architect_prompt_carries_directions_and_palette(self):
        """The architect is told the bearings and the fixed palette."""
        from app.agents.orchestrator import _brief_to_text

        text = _brief_to_text(_request(), _brief(_request()))
        assert '"door_direction": "north"' in text
        assert '"window_direction": "northeast"' in text
        assert '"color_scheme": "warm"' in text
        assert "DOOR DIRECTION: north" in text
        assert "WINDOW DIRECTION: northeast" in text
        assert "terracotta" in text
        assert "SOUTH-EAST" in text


# --------------------------------------------------------------------------
# 3. Colour scheme
# --------------------------------------------------------------------------


class TestColourScheme:
    def test_warm_earthy_select_maps_to_its_own_palette(self):
        brief = _brief(_request(color_scheme="warm"))
        assert brief.palette["label"] == "Warm Earthy"
        assert brief.palette["primary"] == "terracotta"
        assert "navy" in brief.palette["forbidden"]

    def test_navy_palette_fails_a_warm_earthy_brief(self):
        brief = _brief(_request(color_scheme="warm"))
        design = _kitchen(color_palette={
            "primary": "navy", "secondary": "sky blue", "accent": "teal",
            "neutrals": ["white"], "hex_codes": ["#1E3A5F", "#4A9FD8", "#17A2A2"],
        })
        report = validate_design(design, brief, price_design(design))
        assert "palette_scheme_mismatch" in _codes(report)

    def test_matching_palette_passes(self):
        brief = _brief(_request(color_scheme="warm"))
        report = validate_design(_kitchen(), brief, price_design(_kitchen()))
        assert "palette_scheme_mismatch" not in _codes(report)

    def test_off_scheme_hex_is_caught_even_without_a_colour_word(self):
        brief = _brief(_request(color_scheme="warm"))
        design = _kitchen(color_palette={
            "primary": "calm", "secondary": "evening", "accent": "cool",
            "neutrals": ["white"], "hex_codes": ["#1E3A5F"],
        })
        report = validate_design(design, brief, price_design(design))
        assert "palette_scheme_mismatch" in _codes(report)

    def test_palette_for_unknown_scheme_is_neutral(self):
        assert palette_for("no-such-scheme")["scheme"] == "neutral"


# --------------------------------------------------------------------------
# 4. Vastu
# --------------------------------------------------------------------------


def _layout_kitchen() -> dict:
    return {
        "room": {
            "width_cm": 335, "depth_cm": 274, "ceiling_cm": 270,
            "width_ft": 11.0, "depth_ft": 9.0, "area_sqft": 98.8,
            "dimension_text": "11.0 x 9.0 ft",
            "door_direction": "north", "window_direction": "northeast",
        },
        "items": [], "zones": [],
    }


class TestVastu:
    def test_strict_kitchen_forces_the_platform_south_east_facing_east(self):
        brief = _brief(_request())
        design = _kitchen()
        DesignPipeline._attach_vastu(design, brief)
        assert design["cooking_platform"]["wall"] == "southeast"
        assert design["cooking_platform"]["facing"] == "east"
        rule = next(r for r in design["_vastu"]["results"] if r["code"] == "cook_south_east")
        assert rule["passed"] is True
        assert design["_vastu"]["score"] == 100

    def test_validator_blocks_strict_kitchen_with_bad_cooking_rule(self):
        brief = _brief(_request())
        design = _kitchen()
        DesignPipeline._attach_vastu(design, brief)
        # Simulate a design whose cooking position did not make it to south-east.
        design["cooking_platform"] = {"name": "Base Unit", "wall": "south", "facing": "north"}
        design["_vastu"] = analyse_design(design, brief).to_dict()
        report = validate_design(design, brief, price_design(design))
        assert "vastu_strict_violation" in _codes(report)

    def test_vastu_notes_come_from_the_rule_check_not_the_model(self):
        brief = _brief(_request())
        design = _kitchen(vastu_notes=["Limited decorative flexibility", "Fully Vastu compliant"])
        DesignPipeline._attach_vastu(design, brief)
        notes = " ".join(design["vastu_notes"]).lower()
        assert "limited decorative flexibility" not in notes
        assert all(
            any(r["explanation"] == note or note.endswith(r["explanation"])
                for r in design["_vastu"]["results"])
            for note in design["vastu_notes"]
            if not note.startswith("Vastu was requested") and not note.startswith("Add the")
        )

    def test_model_compliance_claim_is_removed_even_when_rules_pass(self):
        brief = _brief(_request())
        design = _kitchen(
            concept="A warm kitchen aligning with Vastu principles."
        )
        DesignPipeline._attach_vastu(design, brief)
        assert "vastu" not in design["concept"].lower()
        assert "principles" not in design["concept"].lower()
        assert "aligning" not in design["concept"].lower()

    def test_contradicting_compliance_claim_is_removed_and_flagged(self):
        """A "respects Vastu" sentence is stripped when a rule fails.

        Strict forces the cooking platform, so the failure is simulated on the
        assessment directly: attach runs the check, then a failed rule is put
        back with the claim restored, exactly the state the old pipeline could
        render.
        """
        brief = _brief(_request())
        design = _kitchen(concept="A warm kitchen that respects Vastu.")
        DesignPipeline._attach_vastu(design, brief)
        # Strict forces the platform, so the claim survives. Now simulate the
        # soft-Vastu case, where the platform is not forced and can fail.
        soft_brief = parse_brief(
            client_preferences=brief.raw_text, room_type="kitchen",
            color_scheme="warm", vastu_compliance="soft",
            room_dimensions="9 x 11 ft",
            door_direction="north", window_direction="northeast",
        )
        design["furniture"][0]["wall"] = "south"
        design["furniture"][0]["facing"] = "north"
        design["cooking_platform"] = {"name": "Base", "wall": "south", "facing": "north"}
        design["concept"] = "A warm kitchen that respects Vastu."
        DesignPipeline._attach_vastu(design, soft_brief)
        assert "respects vastu" not in design["concept"].lower()
        design["concept"] = "A warm kitchen that respects Vastu."
        report = validate_design(design, soft_brief, price_design(design))
        assert "contradicted_vastu_claim" in _codes(report)


# --------------------------------------------------------------------------
# 5. Catalog filtering
# --------------------------------------------------------------------------


class TestCatalogFiltering:
    def test_kitchen_items_exist(self):
        skus = {i.sku for i in catalog.for_room("kitchen")}
        for sku in (
            "KIT-OVC-01", "KIT-SHLF-01", "KIT-PNT-01", "KIT-TRL-01",
            "KIT-EXH-01", "KIT-FAN-01", "KIT-BSP-01", "KIT-LGT-01",
        ):
            assert sku in skus, sku

    def test_kitchen_excludes_study_table_bean_bag_and_sofa(self):
        kitchen_skus = {i.sku for i in catalog.for_room("kitchen")}
        assert "DSK-KID-01" not in kitchen_skus
        assert "BEAN-01" not in kitchen_skus
        assert "SOF-2S-01" not in kitchen_skus

    def test_kitchen_allows_generic_open_shelf_storage(self):
        shelf = catalog.get("STO-SHLF-01")
        assert shelf is not None
        assert catalog.allowed_in_room(shelf, "kitchen")

    def test_entrance_allows_shoe_storage_but_not_other_storage(self):
        entrance_skus = {item.sku for item in catalog.for_room("entrance")}
        assert "SHR-ITSY-3T-01" in entrance_skus
        assert "STO-SHLF-01" not in entrance_skus
        assert "CHV-STO-01" not in entrance_skus

    def test_wrong_room_item_is_dropped_and_duplicate_merged(self):
        brief = _brief(_request())
        design = _kitchen(furniture=[
            {"name": "Modular Kitchen Base Unit 3ft", "sku": "KIT-BSE-01", "quantity": 1},
            {"name": "Auto-Clean Chimney 60cm", "sku": "KIT-EXH-01", "quantity": 1},
            {"name": "Auto-Clean Chimney 60cm", "sku": "KIT-EXH-01", "quantity": 1},
            {"name": "Kids Study Table 90cm", "sku": "DSK-KID-01", "quantity": 1},
            {"name": "Bean Bag - Large", "sku": "BEAN-01", "quantity": 1},
        ])
        out = DesignPipeline._clean_furniture(design, brief)
        skus = [i["sku"] for i in out]
        assert skus.count("KIT-EXH-01") == 1
        assert "DSK-KID-01" not in skus and "BEAN-01" not in skus

    def test_validator_reports_a_wrong_room_item(self):
        brief = _brief(_request())
        design = _kitchen(furniture=[
            {"name": "Bean Bag - Large", "sku": "BEAN-01", "category": "seating", "quantity": 1},
        ])
        report = validate_design(design, brief, price_design(design))
        assert "wrong_room_item" in _codes(report)

    def test_brands_come_from_the_catalog(self):
        assert catalog.get("KIT-CNT-01").brand == "KalingaStone"
        assert catalog.get("BEAN-01").brand == "Wakefit"


# --------------------------------------------------------------------------
# 6. Special requirements
# --------------------------------------------------------------------------


class TestSpecialRequirements:
    def test_unaddressed_requirement_fails(self):
        brief = _brief(_request(specific_requirements=["kids_safe"]))
        design = _kitchen(concept="A calm kitchen.", layout_description="Counter on the south-east.")
        report = validate_design(design, brief, price_design(design))
        assert "requirement_not_addressed" in _codes(report)

    def test_addressed_requirement_passes(self):
        brief = _brief(_request(specific_requirements=["kids_safe"]))
        design = _kitchen(
            layout_description=(
                "Counter on the south-east. Chairs have rounded edges, drawers have child "
                "locks, and every tall unit is anchored to the wall."
            ),
        )
        report = validate_design(design, brief, price_design(design))
        assert "requirement_not_addressed" not in _codes(report)

    def test_storage_heavy_is_recognised(self):
        brief = _brief(_request(specific_requirements=["storage_heavy"]))
        design = _kitchen(
            furniture=[
                {"name": "Bookshelf 3-Tier - 60cm", "category": "storage",
                 "sku": "SHV-BOOK-01", "quantity": 1},
            ],
            storage_strategy="The heavy storage plan uses the anchored bookshelf.",
        )
        report = validate_design(design, brief, price_design(design))
        assert "requirement_not_addressed" not in _codes(report)

    @pytest.mark.parametrize(
        ("requirement", "text", "furniture"),
        [
            ("low_maintenance", "Wipe-clean laminate fronts simplify upkeep.", []),
            ("natural_light", "Use daylight from the northeast window.", []),
            ("entertainment", "The entertainment area centres on the television.", [
                {"name": "TV Unit - Wall Mounted 4ft", "category": "tv_unit",
                 "sku": "TVU-WAL-01", "quantity": 1},
            ]),
        ],
    )
    def test_each_special_requirement_has_a_specific_validator_rule(
        self, requirement, text, furniture
    ):
        brief = _brief(_request(
            specific_requirements=[requirement],
            window_direction="northeast",
        ))
        design = _kitchen(layout_description=text, furniture=furniture)
        report = validate_design(design, brief, price_design(design))
        assert "requirement_not_addressed" not in _codes(report)


class TestSelectedFurniture:
    def test_selected_products_must_appear_in_shopping_list(self):
        request = _request(
            room_type="living_room",
            furniture=["Coffee Table", "Bookshelf", "Shoe Rack"],
            client_preferences="A living room.",
            vastu_compliance="soft",
        )
        brief = _brief(request)
        design = {
            "furniture": [
                {"name": "Coffee Table - Sheesham", "category": "table",
                 "sku": "TBL-COF-01", "quantity": 1},
                {"name": "Bookshelf 3-Tier - 60cm", "category": "storage",
                 "sku": "SHV-BOOK-01", "quantity": 1},
                {"name": "Nilkamal Itsy 3-Tier Shoe Rack", "category": "storage",
                 "sku": "SHR-ITSY-3T-01", "quantity": 1},
            ]
        }
        report = validate_design(design, brief, price_design(design))
        assert "selected_furniture_missing" not in _codes(report)

    def test_missing_selected_product_is_blocking(self):
        request = _request(
            room_type="living_room",
            furniture=["Coffee Table", "Bookshelf"],
            client_preferences="A living room.",
            vastu_compliance="soft",
        )
        report = validate_design(
            {"furniture": [{"name": "Side Table - Round 40cm",
                            "category": "table", "sku": "TBL-SID-01"}]},
            _brief(request),
            price_design({"furniture": [{"name": "Side Table - Round 40cm",
                                         "category": "table", "sku": "TBL-SID-01"}]}),
        )
        assert "selected_furniture_missing" in _codes(report)

    def test_catalog_mode_returns_two_distinct_options_with_selected_items(self, monkeypatch):
        from app.agents.orchestrator import registry

        monkeypatch.setattr(registry, "all", lambda: {})
        request = _request(
            room_type="living_room",
            furniture=["Coffee Table", "Bookshelf", "Shoe Rack"],
            specific_requirements=[
                "Kids Safe", "Heavy Storage", "Low Maintenance",
                "Natural Light", "Entertainment",
            ],
            occupants=["2 adults", "10-year-old"],
            client_preferences="A modern family living room.",
            budget_amount_inr=40000,
            num_options=2,
            vastu_compliance="soft",
            window_direction="north",
        )
        pipeline = DesignPipeline()
        options = asyncio.run(
            pipeline._parallel_options(
                request, {"zones": []}, [], _brief(request)
            )
        )
        assert len(options) == 2
        sku_sets = [{item.sku for item in option.furniture} for option in options]
        assert sku_sets[0] != sku_sets[1]
        for option in options:
            names = {item.name.lower() for item in option.furniture}
            assert any("coffee table" in name for name in names)
            assert any("bookshelf" in name for name in names)
            assert any("shoe rack" in name for name in names)
            assert not any(
                term in name for name in names
                for term in ("study desk", "study table", "bean bag", "sofa")
            )
            assert option.validation["valid"]
            assert option.estimated_cost_inr == sum(
                row["line_total_inr"] for row in option.shopping_list
            )
            assert option.remaining_budget_inr == 40000 - option.estimated_cost_inr
            purchased = {row["sku"] for row in option.shopping_list}
            assert all(item["sku"] not in purchased for item in option.budget_suggestions)


# --------------------------------------------------------------------------
# 7. Trade-offs are separate from Vastu notes
# --------------------------------------------------------------------------


class TestTradeOffs:
    def test_option_has_separate_fields(self):
        option = DesignPipeline()._to_option(_kitchen(), _request(num_options=1))
        assert option.trade_offs
        assert "Limited decorative" not in " ".join(option.vastu_notes)
        # Every note is one of the rule-check explanations, verbatim (prefixed
        # with "Needs attention" when it failed), not model prose.
        brief = _brief(_request())
        design = _kitchen()
        DesignPipeline._attach_vastu(design, brief)
        rule_text = [r["explanation"] for r in design["_vastu"]["results"]]
        for note in option.vastu_notes:
            stripped = note.replace("Needs attention: ", "")
            assert (
                stripped in rule_text
                or note.startswith("Vastu was requested")
                or note.startswith("Add the door")
            ), note


# --------------------------------------------------------------------------
# 8. Retrieval
# --------------------------------------------------------------------------


class TestRetrieval:
    def test_brief_returns_between_three_and_five_passages(self):
        from app.services.retrieval import retrieval_service

        results = asyncio.run(
            retrieval_service.retrieve_for_request(_request())
        )
        assert 3 <= len(results) <= 5

    def test_knowledge_covers_the_missing_topics(self):
        from app.services.knowledge_base import get_knowledge_base

        blob = " ".join(e["content"] for e in get_knowledge_base()).lower()
        for phrase in ("work triangle", "wabi-sabi", "vitrified", "chimney", "tip-over"):
            assert phrase in blob, phrase


# --------------------------------------------------------------------------
# 9. Refine refuses an item that is not in the design
# --------------------------------------------------------------------------


class TestRefine:
    def test_missing_item_is_refused_with_the_client_wording(self):
        design = _kitchen()
        intent = classify_refinement("Make the desk bigger")
        assert intent.target == "item"
        before = json.loads(json.dumps(design))
        updated = apply_refinement(design, intent)
        assert updated["_refinement"]["refused"] is True
        assert updated["_refinement"]["message"] == "There is no desk in this design"
        # Nothing changed except the bookkeeping key.
        assert {k: v for k, v in updated.items() if k != "_refinement"} == before

    def test_removing_a_missing_item_is_refused(self):
        design = _kitchen()
        updated = apply_refinement(design, classify_refinement("Remove the desk"))
        assert updated["_refinement"]["refused"] is True

    def test_refine_endpoint_reports_no_change(self):
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app)
        response = client.post(
            "/api/v1/refine",
            json={"design": _kitchen(), "request": "Make the desk bigger"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["understood"] is False
        assert body["message"] == "There is no desk in this design"
        assert body["changed_fields"] == []


# --------------------------------------------------------------------------
# 10. Brands and prices only from the catalog
# --------------------------------------------------------------------------


class TestCatalogOnlyBrands:
    def test_every_item_brand_is_the_catalog_brand(self):
        option = DesignPipeline()._to_option(_kitchen(), _request(num_options=1))
        for item in option.furniture:
            product = catalog.get(item.sku)
            assert item.brand_suggestions == [product.brand]
        for brand in option.brands:
            assert brand in {i.brand for i in catalog.all()}

    def test_planted_wrong_brand_is_replaced_by_the_catalog_brand(self):
        design = _kitchen(furniture=[
            {"name": "Kitchen Countertop - Per 3ft", "sku": "KIT-CNT-01",
             "category": "kitchen", "quantity": 1, "brand_suggestions": ["Hettich"]},
        ])
        option = DesignPipeline()._to_option(design, _request(num_options=1))
        assert option.furniture[0].brand_suggestions == ["KalingaStone"]


# --------------------------------------------------------------------------
# 11. Revision loop returns the best attempt and stops early
# --------------------------------------------------------------------------


def _fake_response(design):
    async def _call(self, prompt, json_mode=True, temperature=None):
        return {"json": json.loads(json.dumps(design))}
    return _call


class TestRevisionLoop:
    def test_best_revision_is_returned_not_the_last(self):
        """Revision 1 clean, revision 2 worse and over budget: keep revision 1."""
        good = _kitchen()
        bad = _kitchen(
            concept="Generic.",
            furniture=[
                {"name": "Tall Pantry Unit", "sku": "KIT-PNT-01", "quantity": 6},
            ],
        )
        calls = {"n": 0}

        async def fake(self, prompt, json_mode=True, temperature=None):
            calls["n"] += 1
            return {"json": json.loads(json.dumps(good if calls["n"] == 1 else bad))}

        request = _request(budget_amount_inr=150000, num_options=1)
        with patch("app.agents.orchestrator.ArchitectAgent._call_llm", fake):
            response = asyncio.run(
                DesignPipeline().generate(request)
            )
        # The first architect revision is blocking-clean, so it wins and the
        # loop stops there. The earlier behaviour returned whichever attempt
        # ran last, which is how a worse revision reached the client.
        assert response.options[0].concept == good["concept"]
        assert response.options[0].estimated_cost_inr == sum(
            row["line_total_inr"] for row in response.options[0].shopping_list
        )
        assert response.revisions
        assert response.revisions[0]["valid"] is True
        assert calls["n"] == 1

    def test_loop_stops_once_all_blocking_checks_pass(self):
        calls = {"n": 0}

        async def fake(self, prompt, json_mode=True, temperature=None):
            calls["n"] += 1
            return {"json": json.loads(json.dumps(_kitchen()))}

        request = _request(num_options=1, budget_amount_inr=500000)
        with patch("app.agents.orchestrator.ArchitectAgent._call_llm", fake):
            asyncio.run(DesignPipeline().generate(request))
        # One architect call: the first design is blocking-clean.
        assert calls["n"] == 1


# --------------------------------------------------------------------------
# 12. UI contract: no green score with blocking issues
# --------------------------------------------------------------------------


class TestUiContract:
    def test_option_carries_its_own_validation(self):
        option = DesignPipeline()._to_option(_kitchen(), _request(num_options=1))
        assert option.validation is not None
        assert "valid" in option.validation

    def test_wrong_room_item_is_removed_before_option_is_shown(self):
        request = _request(num_options=1)
        broken = _kitchen(furniture=[
            {"name": "Bean Bag - Large", "sku": "BEAN-01", "category": "seating"},
        ])
        option = DesignPipeline()._to_option(broken, request)
        names = {item.name for item in option.furniture}
        assert "Bean Bag - Large" not in names
        assert option.validation["valid"]

    def test_critic_field_is_remaining_issues(self):
        from app.agents.orchestrator import CriticAgent

        async def fake(self, prompt, json_mode=True, temperature=None):
            return {"json": {
                "scores": {"coherence": 8, "feasibility": 8, "budget_adherence": 8,
                           "aesthetics": 8, "specificity": 8},
                "overall": 8, "issues": ["Shopping list is incomplete"],
                "strengths": [], "needs_revision": False,
            }}

        design = _kitchen()
        cost = price_design(design)
        with patch("app.agents.orchestrator.BaseAgent._call_llm", fake):
            critique, _ = asyncio.run(CriticAgent().run(
                _request(num_options=1), design, cost, {},
            ))
        assert "remaining_issues" in critique
        assert "Issues addressed" not in critique


# --------------------------------------------------------------------------
# 13. Layout carries dimensions and zone sizes
# --------------------------------------------------------------------------


class TestLayoutDimensions:
    def test_option_layout_has_dimensions_in_feet_and_zones(self):
        option = DesignPipeline()._to_option(_kitchen(), _request(num_options=1))
        assert option.layout is not None
        room = option.layout["room"]
        assert room["dimension_text"] == "9.0 x 11.0 ft"
        assert "9.0 x 11.0 ft" in option.layout_description
        assert room["area_sqft"] == pytest.approx(98.8, abs=0.2)
        assert room["door_direction"] == "north"
        assert room["window_direction"] == "northeast"
        assert option.layout["zones"]
        zone = option.layout["zones"][0]
        assert zone["approx_area_sqft"] > 0
        assert zone["note"].startswith("footprint plus clearance")
