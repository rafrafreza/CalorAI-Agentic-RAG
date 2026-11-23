"""Domain service: grounded logging, corrections, guards, goals, ownership, memory retrieval."""

from datetime import date, timedelta

import pytest

from backend.db.database import get_db
from backend.db.memory import get_relevant_memories, upsert_memory
from backend.services import nutrition_log as svc
from backend.services.nutrition_log import ItemInput as Item

U = "alice"


def log(*items, **kw):
    return svc.log_meal(U, list(items), **kw)


class TestLogging:
    def test_server_computes_totals_and_description(self):
        r = log(Item(name="paratha", quantity=2), Item(name="chai"), meal_type="breakfast")
        assert r["status"] == "logged"
        meal = r["meal"]
        assert meal["calories"] == 640 and meal["description"] == "2 × Plain Paratha · 1 cup Masala Chai"
        assert all(i["grounded"] and i["kb_id"] for i in meal["items"])
        assert r["day"]["total_calories"] == 640

    def test_kb_values_beat_model_guesses(self):
        r = log(Item(name="paratha", kb_id="paratha", quantity=1, calories=999, protein_g=99, carbs_g=99, fat_g=99))
        assert r["meal"]["calories"] == 260

    def test_unknown_food_requires_labelled_estimate(self):
        assert log(Item(name="beef wellington"))["status"] == "needs_input"
        r = log(Item(name="beef wellington", calories=650, protein_g=35, carbs_g=30, fat_g=42))
        item = r["meal"]["items"][0]
        assert r["status"] == "logged" and not item["grounded"] and item["source"] == "model-estimate"

    def test_material_ambiguity_asks(self):
        r = log(Item(name="dumplings"))
        assert r["status"] == "needs_input" and "ambiguous" in r["problems"][0]

    def test_duplicate_and_implausible_guards(self):
        assert log(Item(name="apple"))["status"] == "logged"
        dup = log(Item(name="apple"))
        assert dup["status"] == "needs_confirmation" and dup["reason"] == "possible_duplicate"
        assert log(Item(name="apple"), confirmed=True)["status"] == "logged"
        big = log(Item(name="paratha", quantity=50))
        assert big["status"] == "needs_confirmation" and big["reason"] == "implausible_quantity"

    def test_past_dates(self):
        r = log(Item(name="roti", quantity=2), day="yesterday")
        assert r["meal"]["date"] == (date.today() - timedelta(days=1)).isoformat()
        assert r["day"]["date"] == r["meal"]["date"]  # totals for the meal's own day
        assert log(Item(name="roti"), day="2999-01-01")["status"] == "error"
        assert log(Item(name="roti"), day="not-a-date")["status"] == "error"


class TestCorrections:
    def test_update_replaces_in_place_no_double_count(self):
        meal_id = log(Item(name="roti", quantity=2), Item(name="dal"))["meal"]["id"]
        r = svc.update_meal(
            U, meal_id, [Item(name="roti", kb_id="roti", quantity=3), Item(name="dal", kb_id="dal_tadka")]
        )
        assert r["status"] == "updated"
        assert r["previous"]["calories"] == 400 and r["meal"]["calories"] == 500
        assert r["meal"]["description"].startswith("3 × Roti")  # description regenerated, never stale
        assert r["day"]["total_calories"] == 500 and r["day"]["meal_count"] == 1

    def test_ownership(self):
        meal_id = log(Item(name="roti"))["meal"]["id"]
        assert svc.update_meal("mallory", meal_id, [Item(name="roti", quantity=9)])["status"] == "error"
        assert svc.delete_meal("mallory", meal_id)["status"] == "error"
        assert svc.delete_meal(U, meal_id)["status"] == "deleted"
        assert svc.delete_meal(U, meal_id)["status"] == "error"  # idempotent: already gone


class TestGoalsAndHistory:
    @pytest.mark.parametrize(
        "key,value,expected",
        [
            ("protein_target", "140g daily", {"protein_g": 140}),
            ("calorie_goal", "stay under 2000 kcal", {"calories": 2000}),
            ("daily_target", "2200 calories and 120 g protein", {"calories": 2200, "protein_g": 120}),
        ],
    )
    def test_parse_targets(self, key, value, expected):
        assert svc.parse_targets([{"key": key, "value": value, "category": "target"}]) == expected

    def test_day_summary_includes_remaining(self):
        with get_db() as conn:
            upsert_memory(conn, U, "protein_target", "140g daily", "target")
        log(Item(name="chicken breast", quantity=200, unit="g"))
        with get_db() as conn:
            day = svc.day_summary(conn, U, svc.parse_day("today"))
        assert day["remaining"]["protein_g"] == pytest.approx(78)

    def test_search_history(self):
        log(Item(name="paneer butter masala"), Item(name="naan", quantity=2))
        hits = svc.search_history(U, "that paneer curry")
        assert hits and "Paneer Butter Masala" in hits[0]["description"]


class TestMemoryRetrieval:
    def test_preferences_are_pinned_even_when_unrelated(self):
        with get_db() as conn:
            upsert_memory(conn, U, "allergy_peanuts", "allergic to peanuts", "preference")
            upsert_memory(conn, U, "usual_breakfast", "2 parathas and chai", "shortcut")
            upsert_memory(conn, U, "sibling", "brother often shares meals", "personal")
            facts = get_relevant_memories(conn, U, "had a samosa")
        keys = {f["key"] for f in facts}
        assert "allergy_peanuts" in keys  # pinned: safety-critical facts are always in context
        assert "sibling" not in keys  # unrelated long-tail fact stays out of the prompt

    def test_reference_cues_pull_in_shortcuts(self):
        with get_db() as conn:
            upsert_memory(conn, U, "usual_breakfast", "2 parathas and chai", "shortcut")
            facts = get_relevant_memories(conn, U, "my usual please")
        assert [f["key"] for f in facts] == ["usual_breakfast"]
