"""
Meal-logging domain service - "the LLM proposes, the engine disposes".

The model decides *what* was eaten (items, quantities, which KB entry). This module
decides *how much that is*:

* every item is resolved against the knowledge base; grounded numbers always win
  over model guesses, and unknown foods are accepted only as labelled estimates,
* macros are summed in Python (LLMs are bad at arithmetic),
* the meal description is generated from the items, so a correction can never
  leave a stale "2 rotis" description on a 3-roti meal (a v1 prompt-level bug class),
* duplicate and implausible logs are *structurally* held for confirmation instead
  of relying on prompt instructions.

Tools, the offline agent and the REST API all go through these functions.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from pydantic import BaseModel, Field, field_validator

from backend.db import meals as meal_db
from backend.db.database import get_db
from backend.db.memory import get_all_memory
from backend.rag.corpus import FoodDoc
from backend.rag.portion import COUNT_UNITS, MASS_G, ScaledNutrition, scale
from backend.rag.profile import DietProfile, profile_from_memories
from backend.rag.quantity import UNIT_INDEX
from backend.rag.service import get_rag

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack", "other")
MAX_ITEM_GRAMS = 2500
MAX_COUNT = 25
MAX_MEAL_KCAL = 4000
DUPLICATE_WINDOW_S = 180


class ItemInput(BaseModel):
    """One food item proposed by the model (or parsed by the offline agent)."""

    name: str = Field(description="The food as the user described it, e.g. 'paratha' or 'masala chai'.")
    quantity: float | None = Field(
        None, description="Amount in `unit`. Omit when the user gave no amount (one serving is assumed)."
    )
    unit: str | None = Field(
        None,
        description="piece, cup, bowl, plate, glass, slice, serving, box, g, ml, tbsp ... "
        "Omit for the food's natural unit (pieces of roti, cups of chai).",
    )
    kb_id: str | None = Field(
        None, description="Knowledge-base id from the retrieved context or lookup_nutrition. Strongly preferred."
    )
    calories: float | None = Field(
        None, description="ONLY for foods NOT in the knowledge base: your kcal estimate for this item at this quantity."
    )
    protein_g: float | None = Field(None, description="Estimate; only together with `calories`.")
    carbs_g: float | None = Field(None, description="Estimate; only together with `calories`.")
    fat_g: float | None = Field(None, description="Estimate; only together with `calories`.")
    fiber_g: float | None = Field(None, description="Estimate; optional.")

    @field_validator("quantity")
    @classmethod
    def _positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("quantity must be positive")
        return v

    @field_validator("unit")
    @classmethod
    def _canonical_unit(cls, v: str | None) -> str | None:
        if not v:
            return None
        v = v.strip().lower()
        return UNIT_INDEX.get(v, v)


# ── helpers ──────────────────────────────────────────────────────────────────


def parse_day(value: str | None) -> str:
    today = date.today()
    v = (value or "").strip().lower()
    if v in ("", "today", "now", "tonight"):
        return today.isoformat()
    if v == "yesterday":
        return (today - timedelta(days=1)).isoformat()
    if v == "day before yesterday":
        return (today - timedelta(days=2)).isoformat()
    try:
        d = date.fromisoformat(v[:10])
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD, 'today' or 'yesterday'.") from exc
    if d > today:
        raise ValueError("Meals can't be logged in the future.")
    if d < today - timedelta(days=366):
        raise ValueError("That date is more than a year ago.")
    return d.isoformat()


def infer_meal_type(now: datetime | None = None) -> str:
    h = (now or datetime.now()).hour
    if 5 <= h < 11:
        return "breakfast"
    if 11 <= h < 16:
        return "lunch"
    if 16 <= h < 19:
        return "snack"
    if 19 <= h < 23:
        return "dinner"
    return "snack"


def load_profile(user_id: str) -> DietProfile:
    with get_db() as conn:
        return profile_from_memories(get_all_memory(conn, user_id))


def _fmt_qty(q: float) -> str:
    return f"{q:.2f}".rstrip("0").rstrip(".")


def item_label(item: dict) -> str:
    q, unit, name = item.get("quantity"), item.get("unit"), item["name"]
    if q is None:
        return name
    if unit in (None, "piece"):
        return f"{_fmt_qty(q)} × {name}"
    return f"{_fmt_qty(q)} {unit} {name}"


def describe(items: list[dict]) -> str:
    return " · ".join(item_label(i) for i in items)


def _display_quantity(doc: FoodDoc, item: ItemInput, n: ScaledNutrition) -> tuple[float, str]:
    if item.unit:
        return (item.quantity if item.quantity is not None else 1.0, item.unit)
    if doc.serving_unit in COUNT_UNITS:
        return (item.quantity if item.quantity is not None else doc.serving_qty, doc.serving_unit)
    if doc.serving_unit in MASS_G and doc.piece_g and item.quantity is not None:
        return (item.quantity, "piece")
    return (round(n.factor * doc.serving_qty, 2), doc.serving_unit)


# ── item resolution ──────────────────────────────────────────────────────────


def resolve_items(items: list[ItemInput], profile: DietProfile | None = None) -> tuple[list[dict], list[str]]:
    """Ground each item in the KB (or accept a labelled estimate). Returns (items, problems)."""
    rag = get_rag()
    resolved: list[dict] = []
    problems: list[str] = []
    for it in items:
        if it.unit and it.quantity is None:
            it = it.model_copy(update={"quantity": 1.0})
        doc = rag.get(it.kb_id) if it.kb_id else None
        confidence: float | None = 1.0 if doc else None
        if doc is None:
            g = rag.ground(it.name, it.quantity, it.unit, profile)
            if g.usable:
                doc, confidence = g.match, g.confidence
            elif g.status == "ambiguous" and g.match and it.calories is None:
                options = ", ".join([g.match.name] + [a["name"] for a in g.alternatives[:2]])
                problems.append(
                    f"'{it.name}' is ambiguous ({options}) and the calorie difference matters. "
                    "Ask the user which one they had, or pass the matching kb_id."
                )
                continue

        if doc is not None:
            try:
                n = scale(doc, it.quantity, it.unit)
            except ValueError as exc:
                problems.append(f"'{it.name}': {exc}")
                continue
            qty, unit = _display_quantity(doc, it, n)
            resolved.append(
                {
                    "name": doc.name,
                    "query": it.name,
                    "kb_id": doc.id,
                    "quantity": qty,
                    "unit": unit,
                    "grams": n.grams,
                    "calories": n.kcal,
                    "protein_g": n.protein_g,
                    "carbs_g": n.carbs_g,
                    "fat_g": n.fat_g,
                    "fiber_g": n.fiber_g,
                    "grounded": True,
                    "source": doc.source,
                    "confidence": round(confidence or 0, 3),
                    "assumptions": n.assumptions,
                    "conflicts": profile.conflicts(doc) if profile else [],
                }
            )
        elif it.calories is not None:
            resolved.append(
                {
                    "name": it.name.strip().title(),
                    "query": it.name,
                    "kb_id": None,
                    "quantity": it.quantity if it.quantity is not None else 1.0,
                    "unit": it.unit or "serving",
                    "grams": None,
                    "calories": round(max(0.0, it.calories), 1),
                    "protein_g": round(max(0.0, it.protein_g or 0.0), 1),
                    "carbs_g": round(max(0.0, it.carbs_g or 0.0), 1),
                    "fat_g": round(max(0.0, it.fat_g or 0.0), 1),
                    "fiber_g": round(max(0.0, it.fiber_g or 0.0), 1),
                    "grounded": False,
                    "source": "model-estimate",
                    "confidence": None,
                    "assumptions": ["Not in the knowledge base - model estimate."],
                    "conflicts": [],
                }
            )
        else:
            problems.append(
                f"'{it.name}' is not in the knowledge base. Either ask the user to clarify, or call again with "
                "your own calories/protein_g/carbs_g/fat_g estimate for this item (it will be labelled as an estimate)."
            )
    return resolved, problems


def sum_items(items: list[dict]) -> dict:
    keys = ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")
    return {k: round(sum(float(i.get(k) or 0) for i in items), 1) for k in keys}


def plausibility_flags(items: list[dict]) -> list[str]:
    flags = []
    for i in items:
        if i.get("grams") and i["grams"] > MAX_ITEM_GRAMS:
            flags.append(f"{item_label(i)} is about {i['grams']:.0f} g - unusually large.")
        elif i.get("unit") in COUNT_UNITS and (i.get("quantity") or 0) > MAX_COUNT:
            flags.append(f"{item_label(i)} is an unusually high count.")
    total = sum_items(items)["calories"]
    if total > MAX_MEAL_KCAL:
        flags.append(f"The meal totals {total:.0f} kcal - unusually high for one meal.")
    return flags


def _notes(items: list[dict]) -> list[str]:
    notes: list[str] = []
    for i in items:
        for a in i.get("assumptions") or []:
            notes.append(f"{i['name']}: {a}")
        for c in i.get("conflicts") or []:
            notes.append(f"⚠ {c}")
    return list(dict.fromkeys(notes))


# ── targets & summaries ──────────────────────────────────────────────────────

_NUM = r"(\d{2,5}(?:\.\d+)?)"
_MACRO_WORDS = {
    "calories": r"(?:kcal|kcals|cal|cals|calorie|calories)",
    "protein_g": r"protein",
    "carbs_g": r"(?:carb|carbs|carbohydrates?)",
    "fat_g": r"(?:fat|fats)",
}
_RANGES = {"calories": (800, 6000), "protein_g": (20, 400), "carbs_g": (20, 900), "fat_g": (10, 300)}


def parse_targets(memories: list[dict]) -> dict:
    """Extract numeric daily goals from 'target' memories ("protein_target: 140g daily")."""
    targets: dict[str, float] = {}
    for m in memories:
        if m.get("category") != "target" and not re.search(r"goal|target", m.get("key", "")):
            continue
        text = f"{m.get('key', '').replace('_', ' ')} {m.get('value', '')}".lower()
        for macro, word in _MACRO_WORDS.items():
            hit = re.search(rf"{_NUM}\s*(?:g|grams?)?\s*(?:of\s+)?{word}\b", text) or re.search(
                rf"\b{word}\w*\s*(?:target|goal|intake|budget|limit)?\s*(?:of|is|:|=|-|at|under|below|around|about|to)?\s*{_NUM}",
                text,
            )
            if hit:
                val = float(hit.group(1))
                lo, hi = _RANGES[macro]
                if lo <= val <= hi:
                    targets[macro] = val
    return targets


def day_summary(conn, user_id: str, day: str) -> dict:
    totals = meal_db.get_daily_totals(conn, user_id, day)
    targets = parse_targets(get_all_memory(conn, user_id))
    key_map = {
        "calories": "total_calories",
        "protein_g": "total_protein_g",
        "carbs_g": "total_carbs_g",
        "fat_g": "total_fat_g",
    }
    remaining = {k: round(v - totals[key_map[k]], 1) for k, v in targets.items()}
    return {**totals, "targets": targets, "remaining": remaining}


def public_meal(meal: dict) -> dict:
    keys = (
        "id",
        "timestamp",
        "meal_type",
        "description",
        "calories",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fiber_g",
        "items",
        "source",
    )
    out = {k: meal.get(k) for k in keys}
    out["date"] = str(meal.get("timestamp", ""))[:10]
    return out


# ── commands ─────────────────────────────────────────────────────────────────


def log_meal(
    user_id: str,
    items: list[ItemInput],
    meal_type: str | None = None,
    day: str | None = None,
    source: str = "text",
    raw_input: str | None = None,
    image_path: str | None = None,
    confirmed: bool = False,
) -> dict:
    if not items:
        return {"status": "error", "error": "No items to log."}
    try:
        day = parse_day(day)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}
    meal_type = (meal_type or "").lower() or None
    if meal_type not in MEAL_TYPES:
        meal_type = infer_meal_type()

    resolved, problems = resolve_items(items, load_profile(user_id))
    if problems:
        return {"status": "needs_input", "problems": problems, "resolved": [item_label(i) for i in resolved]}

    with get_db() as conn:
        if not confirmed:
            dup = meal_db.find_recent_duplicate(
                conn, user_id, day, meal_db.item_signature(resolved), DUPLICATE_WINDOW_S
            )
            if dup:
                return {
                    "status": "needs_confirmation",
                    "reason": "possible_duplicate",
                    "existing_meal_id": dup["id"],
                    "message": f"An identical meal ({dup['description']}) was logged moments ago. Ask whether this is "
                    "a second serving or a duplicate message; only if confirmed, call again with confirmed=true.",
                }
            flags = plausibility_flags(resolved)
            if flags:
                return {
                    "status": "needs_confirmation",
                    "reason": "implausible_quantity",
                    "flags": flags,
                    "message": "This looks like a possible typo. Confirm with the user; only if confirmed, "
                    "call again with confirmed=true.",
                }
        totals = sum_items(resolved)
        meal_id = meal_db.create_meal(
            conn,
            user_id,
            description=describe(resolved),
            items=resolved,
            meal_type=meal_type,
            source=source,
            raw_input=raw_input,
            image_path=image_path,
            target_date=day,
            **{
                "calories": totals["calories"],
                "protein_g": totals["protein_g"],
                "carbs_g": totals["carbs_g"],
                "fat_g": totals["fat_g"],
                "fiber_g": totals["fiber_g"],
            },
        )
        meal = meal_db.get_meal_by_id(conn, meal_id)
        summary = day_summary(conn, user_id, day)
    return {"status": "logged", "meal": public_meal(meal), "day": summary, "notes": _notes(resolved)}


def update_meal(
    user_id: str,
    meal_id: int,
    items: list[ItemInput] | None = None,
    meal_type: str | None = None,
    confirmed: bool = False,
) -> dict:
    with get_db() as conn:
        existing = meal_db.get_meal_by_id(conn, meal_id, user_id)
    if not existing:
        return {"status": "error", "error": f"Meal {meal_id} not found. Use get_meals or search_meal_history first."}

    fields: dict = {}
    notes: list[str] = []
    if items:
        resolved, problems = resolve_items(items, load_profile(user_id))
        if problems:
            return {"status": "needs_input", "problems": problems}
        flags = plausibility_flags(resolved)
        if flags and not confirmed:
            return {
                "status": "needs_confirmation",
                "reason": "implausible_quantity",
                "flags": flags,
                "message": "Confirm with the user; only if confirmed, call again with confirmed=true.",
            }
        fields.update(items=resolved, description=describe(resolved), **sum_items(resolved))
        notes = _notes(resolved)
    if meal_type:
        if meal_type.lower() not in MEAL_TYPES:
            return {"status": "error", "error": f"meal_type must be one of {MEAL_TYPES}"}
        fields["meal_type"] = meal_type.lower()
    if not fields:
        return {"status": "error", "error": "Nothing to update: pass the full corrected `items` list and/or meal_type."}

    with get_db() as conn:
        meal_db.update_meal(conn, meal_id, user_id=user_id, **fields)
        meal = meal_db.get_meal_by_id(conn, meal_id, user_id)
        summary = day_summary(conn, user_id, str(meal["timestamp"])[:10])
    return {
        "status": "updated",
        "meal": public_meal(meal),
        "previous": {"description": existing["description"], "calories": existing["calories"]},
        "day": summary,
        "notes": notes,
    }


def delete_meal(user_id: str, meal_id: int) -> dict:
    with get_db() as conn:
        meal = meal_db.get_meal_by_id(conn, meal_id, user_id)
        if not meal or not meal_db.delete_meal(conn, meal_id, user_id):
            return {"status": "error", "error": f"Meal {meal_id} not found."}
        summary = day_summary(conn, user_id, str(meal["timestamp"])[:10])
    return {"status": "deleted", "meal": public_meal(meal), "day": summary}


def search_history(user_id: str, query: str, days: int = 30, limit: int = 5) -> list[dict]:
    """Fuzzy + lexical search over the user's own meals ("that paneer wrap from last week")."""
    from backend.rag.indexes import CharNgramIndex
    from backend.rag.text import term_views, view_similarity

    since = (date.today() - timedelta(days=max(1, min(days, 365)))).isoformat()
    with get_db() as conn:
        meals = meal_db.get_meals_since(conn, user_id, since)
    if not meals or not query.strip():
        return []
    texts = []
    for idx, m in enumerate(meals):
        texts.append((idx, m["description"]))
        texts.extend((idx, i.get("name", "")) for i in m["items"])
        texts.extend((idx, i.get("query", "")) for i in m["items"] if i.get("query"))
    index = CharNgramIndex(texts)
    fuzzy = dict(index.search(query, top_k=len(meals)))
    q_views = term_views(query)
    scored = []
    for idx, m in enumerate(meals):
        vocab = term_views(" ".join([m["description"]] + [i.get("query", "") for i in m["items"]]))
        lexical = (
            (sum(max((view_similarity(q, v) for v in vocab), default=0) for q in q_views) / len(q_views))
            if q_views
            else 0
        )
        score = 0.5 * fuzzy.get(idx, 0.0) + 0.5 * lexical
        if score >= 0.35:
            scored.append((score, m))
    scored.sort(key=lambda x: -x[0])
    return [{**public_meal(m), "score": round(s, 3)} for s, m in scored[:limit]]
