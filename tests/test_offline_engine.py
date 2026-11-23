"""The deterministic offline engine (no LLM): facts, corrections, totals, guards."""

from backend.agent import offline
from backend.db.database import get_db
from backend.db.memory import get_all_memory
from backend.rag.service import get_rag

U = "offline-user"


def turn(text: str):
    groundings = get_rag().ground_message(text)
    return offline.respond(U, text, groundings, meal_type=None, raw_input=text)


def test_captures_every_goal_and_the_diet_in_one_message():
    reply, actions = turn("I'm vegetarian, my goal is 2000 kcal and 120g protein")
    with get_db() as conn:
        facts = {m["key"]: m["value"] for m in get_all_memory(conn, U)}
    assert facts == {
        "dietary_restriction": "vegetarian",
        "calorie_target": "2000 kcal daily",
        "protein_target": "120 g protein daily",
    }
    assert "remember" in reply and len(actions) == 3


def test_allergy():
    turn("I'm allergic to peanuts")
    with get_db() as conn:
        assert get_all_memory(conn, U)[0]["value"] == "allergic to peanuts"


def test_log_correct_with_units_and_delete():
    _, actions = turn("a bowl of dal and 2 rotis")
    assert actions[0]["result"]["meal"]["calories"] == 400
    _, actions = turn("actually it was 2 bowls of dal not 1")
    meal = actions[0]["result"]["meal"]
    assert meal["calories"] == 600 and "2 bowl Dal Tadka" in meal["description"]
    _, actions = turn("delete the last one")
    assert actions[0]["result"]["status"] == "deleted"


def test_unknown_and_vague_inputs_never_log():
    for text in ["beef wellington", "skipped lunch but grazed all afternoon", "hello"]:
        _, actions = turn(text)
        assert not any(a["tool"] == "log_meal" for a in actions), text


def test_totals_without_meals():
    reply, _ = turn("how am I doing?")
    assert "Nothing logged yet" in reply
