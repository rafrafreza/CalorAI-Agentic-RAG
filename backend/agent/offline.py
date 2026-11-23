"""
Offline engine: a deterministic, rule-based agent built on the same RAG + domain
services as the LLM agent.

It exists for three reasons:

1. **Resilience** - if the LLM provider is down or rate-limited *before* any tool has
   run, simple turns (logging, totals, corrections, preferences) still work.
2. **Zero-key try-out** - anyone can clone the repo and use the app without an API key.
3. **Testability** - CI exercises real end-to-end flows through the HTTP API.

It is intentionally narrow; anything it doesn't understand gets a helpful reply,
never a guess.
"""

from __future__ import annotations

import re

from backend.db.database import get_db
from backend.db.meals import get_meals
from backend.db.memory import normalize_key, upsert_memory
from backend.rag.quantity import NUMBER_WORDS, parse_quantity
from backend.rag.service import Grounding, get_rag
from backend.services import nutrition_log as svc
from backend.services.nutrition_log import ItemInput

# In offline mode the UI/API already expose mode="offline"; only an unexpected outage is called out inline.
OFFLINE_NOTE = ""
DEGRADED_NOTE = "\n\n_The AI model is unreachable right now, so I used the offline engine._"

_GREETING = re.compile(r"^\s*(hi|hello|hey|yo|hola|namaste|good (morning|afternoon|evening))\b", re.I)
_TOTALS = re.compile(
    r"how am i doing|how'?s my day|how is my day|so far|summary|progress|totals?\b|remaining|left today|"
    r"how much (protein|calories|carbs|fat)|calories today|protein today",
    re.I,
)
_DIET = re.compile(r"\bi(?:'?m| am)\s+(?:a |an )?(?:strict |pure )?(vegetarian|vegan|pescatarian|eggetarian)\b", re.I)
_ALLERGY = re.compile(r"\b(?:allergic to|allergy to|intolerant to)\s+([a-z ]{3,30}?)(?:[.,!]|$| and | but )", re.I)
_GOAL = re.compile(r"\b(?:goal|target|aim|trying to (?:hit|eat|get)|want to (?:hit|eat|get))\b", re.I)
_AMOUNT = re.compile(r"(\d{2,5})\s*(?:g|grams)?\s*(?:of\s+)?(protein|kcal|calories|cals?)\b", re.I)
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(k for k in NUMBER_WORDS if k not in {"a", "an"}) + r")"
_CORRECTION = re.compile(
    r"\b(?:actually|no|oops|sorry|correction|make (?:that|it)|i meant|it was)\b.*?"
    rf"{_NUM}\s+([a-z][a-z ]*?)\s+(?:not|instead of)\s+{_NUM}\b",
    re.I,
)
_DELETE = re.compile(r"\b(delete|remove|undo|scratch)\b.*\b(last|that|previous)\b", re.I)


def _num(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        return float(NUMBER_WORDS.get(tok.lower(), 1))


def _fmt_day(day: dict) -> str:
    line = f"Today: **{day['total_calories']:.0f}"
    if "calories" in day.get("targets", {}):
        line += f" / {day['targets']['calories']:.0f}"
    line += f" kcal** · {day['total_protein_g']:.0f} g protein"
    if "protein_g" in day.get("targets", {}):
        line += f" (goal {day['targets']['protein_g']:.0f} g)"
    return line + f" · {day['meal_count']} meal{'s' if day['meal_count'] != 1 else ''}."


def _logged_reply(result: dict) -> str:
    meal = result["meal"]
    est = [i["name"] for i in meal["items"] if not i.get("grounded")]
    text = (
        f"Logged **{meal['description']}** — **{meal['calories']:.0f} kcal** "
        f"({meal['protein_g']:.0f} g P · {meal['carbs_g']:.0f} g C · {meal['fat_g']:.0f} g F).\n"
        f"{_fmt_day(result['day'])}"
    )
    warnings = [n for n in result.get("notes", []) if n.startswith("⚠")]
    if warnings:
        text += "\n" + "\n".join(warnings)
    if est:
        text += f"\n_Estimated (not in the knowledge base): {', '.join(est)}._"
    return text


def respond(
    user_id: str,
    text: str,
    groundings: list[Grounding],
    meal_type: str | None,
    raw_input: str | None = None,
    degraded: bool = False,
) -> tuple[str, list[dict]]:
    """Return (markdown reply, actions) for one turn."""
    note = DEGRADED_NOTE if degraded else OFFLINE_NOTE
    actions: list[dict] = []
    msg = text.strip()

    # 1. durable facts
    facts = []
    if m := _DIET.search(msg):
        facts.append(("dietary_restriction", m.group(1).lower(), "preference"))
    for m in _ALLERGY.finditer(msg + "."):
        thing = m.group(1).strip().lower()
        facts.append((f"allergy_{normalize_key(thing)}", f"allergic to {thing}", "preference"))
    if goal := _GOAL.search(msg):
        for m in _AMOUNT.finditer(msg, goal.start()):  # "goal is 2000 kcal and 120 g protein"
            amount, what = m.group(1), m.group(2).lower()
            key = "protein_target" if what == "protein" else "calorie_target"
            facts.append((key, f"{amount} g protein daily" if what == "protein" else f"{amount} kcal daily", "target"))
    if facts:
        with get_db() as conn:
            for key, value, cat in facts:
                upsert_memory(conn, user_id, key, value, cat)
                actions.append(
                    {
                        "tool": "set_user_memory",
                        "result": {"status": "remembered", "key": key, "value": value, "category": cat},
                    }
                )
        listed = "; ".join(v for _, v, _ in facts)
        return (
            f"Noted — I'll remember: **{listed}**. I'll use it to pick the right foods and flag conflicts." + note,
            actions,
        )

    # 2. corrections: "actually that was 3 rotis not 2"
    if m := _CORRECTION.search(msg):
        parsed = parse_quantity(f"{m.group(1)} {m.group(2).strip()}")
        new_qty, new_unit, food = parsed.quantity or _num(m.group(1)), parsed.unit, parsed.food or m.group(2).strip()
        target = get_rag().ground(food)
        if target.match:
            with get_db() as conn:
                todays = get_meals(conn, user_id, target_date=svc.parse_day("today"))
            for meal in reversed(todays):
                if any(i.get("kb_id") == target.match.id for i in meal["items"]):
                    items = [
                        ItemInput(
                            name=i["query"] or i["name"],
                            kb_id=i.get("kb_id"),
                            quantity=new_qty if i.get("kb_id") == target.match.id else i.get("quantity"),
                            unit=new_unit if i.get("kb_id") == target.match.id else i.get("unit"),
                            calories=None if i.get("grounded") else i.get("calories"),
                            protein_g=None if i.get("grounded") else i.get("protein_g"),
                            carbs_g=None if i.get("grounded") else i.get("carbs_g"),
                            fat_g=None if i.get("grounded") else i.get("fat_g"),
                        )
                        for i in meal["items"]
                    ]
                    result = svc.update_meal(user_id, meal["id"], items)
                    actions.append({"tool": "update_meal", "result": result})
                    if result["status"] == "updated":
                        return (
                            f"Fixed it — updated to **{result['meal']['description']}** "
                            f"({result['previous']['calories']:.0f} → **{result['meal']['calories']:.0f} kcal**). "
                            f"No double counting.\n{_fmt_day(result['day'])}" + note
                        ), actions
                    break
        return "I couldn't find that item in today's meals to correct. Which meal was it?" + note, actions

    # 3. delete last
    if _DELETE.search(msg):
        with get_db() as conn:
            todays = get_meals(conn, user_id, target_date=svc.parse_day("today"))
        if not todays:
            return "There's nothing logged today to remove." + note, actions
        result = svc.delete_meal(user_id, todays[-1]["id"])
        actions.append({"tool": "delete_meal", "result": result})
        return f"Removed **{todays[-1]['description']}**.\n{_fmt_day(result['day'])}" + note, actions

    usable = [g for g in groundings if g.usable]

    # 4. totals / progress
    if _TOTALS.search(msg) and not usable:
        with get_db() as conn:
            day = svc.day_summary(conn, user_id, svc.parse_day("today"))
        actions.append({"tool": "get_daily_totals", "result": day})
        if day["meal_count"] == 0:
            return (
                "Nothing logged yet today — tell me what you've eaten and I'll keep the running total." + note,
                actions,
            )
        rem = day.get("remaining", {})
        extra = f"\nRemaining: **{rem['calories']:.0f} kcal**" if "calories" in rem else ""
        if "protein_g" in rem:
            extra += f"{',' if extra else chr(10) + 'Remaining:'} **{rem['protein_g']:.0f} g protein**"
        return _fmt_day(day) + extra + note, actions

    # 5. greeting
    if _GREETING.match(msg) and not usable:
        return (
            'Hey! Tell me what you ate — e.g. *"2 parathas and chai for breakfast"* — '
            "and I'll log it with grounded numbers." + note
        ), actions

    # 6. logging
    ambiguous = [g for g in groundings if g.status == "ambiguous" and g.material_ambiguity]
    unknown = [g for g in groundings if g.status == "no_match"]
    if ambiguous and not usable:
        g = ambiguous[0]
        options = " or ".join([g.match.name] + [a["name"] for a in g.alternatives[:2]])
        return f"Quick check — was the *{g.query}* {options}?" + note, actions
    if usable:
        items = [ItemInput(name=g.query, kb_id=g.match.id, quantity=g.quantity, unit=g.unit) for g in usable]
        result = svc.log_meal(user_id, items, meal_type=meal_type, raw_input=raw_input)
        actions.append({"tool": "log_meal", "result": result})
        if result["status"] == "logged":
            reply = _logged_reply(result)
            if unknown:
                reply += f"\nI couldn't find *{', '.join(g.query for g in unknown)}* — can you describe it differently?"
            return reply + note, actions
        if result["status"] == "needs_confirmation":
            return (
                "That looks like "
                + (
                    "a duplicate of a meal you just logged"
                    if result["reason"] == "possible_duplicate"
                    else "an unusually large amount"
                )
                + " — did you mean it? Reply with the corrected amount."
                + note
            ), actions
        return " ".join(result.get("problems", ["I couldn't log that."])) + note, actions
    if unknown:
        return (
            f"I don't have *{', '.join(g.query for g in unknown)}* in my nutrition knowledge base yet. "
            "Could you describe it with simpler ingredients?" + note
        ), actions

    return (
        'I can log meals (*"a bowl of dal and 2 rotis"*), show your totals (*"how am I doing?"*), '
        'fix entries (*"actually 3 rotis not 2"*) and remember preferences (*"I\'m vegetarian"*).' + note
    ), actions
