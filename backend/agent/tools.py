"""
Agent tools - thin, typed adapters over the domain services.

Boundaries (non-overlapping, one responsibility each):

    lookup_nutrition     READ   hybrid-RAG grounding of one food (KB id, macros, citations)
    log_meal             CREATE one meal; server grounds items and computes totals
    update_meal          UPDATE replace a meal's items (corrections - never re-log)
    delete_meal          DELETE soft delete
    get_meals            READ   meals by date / last N / meal type
    search_meal_history  READ   fuzzy search over the user's own past meals
    get_daily_totals     READ   aggregate + goals + remaining
    get_user_memory      READ   long-term facts
    set_user_memory      WRITE  long-term fact
    delete_user_memory   DELETE long-term fact

The acting user comes from the run config (``configurable.user_id``), never from
the model, so one user's agent can never touch another user's data.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from backend.db.database import get_db
from backend.db.meals import get_meals as db_get_meals
from backend.db.memory import (
    CATEGORIES,
    delete_memory,
    generate_embedding,
    get_all_memory,
    get_relevant_memories,
    normalize_key,
    upsert_memory,
)
from backend.rag.service import get_rag
from backend.services import nutrition_log as svc
from backend.services.nutrition_log import ItemInput


def _cfg(config: RunnableConfig | None) -> dict:
    return (config or {}).get("configurable", {}) or {}


def _user(config: RunnableConfig | None) -> str:
    return _cfg(config).get("user_id") or "default"


def _json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _compact_meal(m: dict) -> dict:
    return {
        "meal_id": m["id"],
        "date": str(m.get("timestamp", ""))[:16],
        "meal_type": m.get("meal_type"),
        "description": m["description"],
        "calories": m["calories"],
        "items": [
            {"name": i.get("name"), "kb_id": i.get("kb_id"), "quantity": i.get("quantity"), "unit": i.get("unit")}
            for i in m.get("items", [])
        ],
    }


# ── nutrition ────────────────────────────────────────────────────────────────


class LookupArgs(BaseModel):
    food: str = Field(description="One food, e.g. 'paneer tikka' (one call per food).")
    quantity: float | None = Field(None, description="Amount in `unit`, if known.")
    unit: str | None = Field(None, description="piece, cup, bowl, plate, glass, g, ml ... if known.")


@tool(args_schema=LookupArgs)
def lookup_nutrition(
    food: str, quantity: float | None = None, unit: str | None = None, config: RunnableConfig = None
) -> str:
    """Look up ONE food in the nutrition knowledge base (hybrid BM25 + phonetic + semantic retrieval).
    Returns status (grounded | ambiguous | no_match), kb_id, nutrition for the quantity, alternatives,
    and diet/allergen conflicts for this user. Use for foods not already in the retrieved context."""
    try:
        g = get_rag().ground(food, quantity, unit, svc.load_profile(_user(config)))
        return _json(g.to_llm())
    except Exception as exc:  # tools must always answer the model
        return _json({"status": "error", "error": str(exc)})


class LogMealArgs(BaseModel):
    items: list[ItemInput] = Field(description="Every item in this ONE meal.")
    meal_type: Literal["breakfast", "lunch", "dinner", "snack"] | None = Field(
        None, description="Omit to infer from the time of day."
    )
    date: str | None = Field(None, description="'today' (default), 'yesterday' or YYYY-MM-DD for past meals.")
    confirmed: bool = Field(
        False,
        description="True ONLY after the user explicitly confirmed a flagged duplicate or an unusually large amount.",
    )


@tool(args_schema=LogMealArgs)
def log_meal(
    items: list[ItemInput],
    meal_type: str | None = None,
    date: str | None = None,
    confirmed: bool = False,
    config: RunnableConfig = None,
) -> str:
    """Log ONE meal with all of its items. The server grounds each item in the knowledge base and computes
    every total - never pre-compute numbers. Returns the saved meal and the day's running totals."""
    c = _cfg(config)
    try:
        return _json(
            svc.log_meal(
                _user(config),
                items,
                meal_type=meal_type,
                day=date,
                source=c.get("source", "text"),
                raw_input=c.get("raw_input"),
                image_path=c.get("image_path"),
                confirmed=confirmed,
            )
        )
    except Exception as exc:
        return _json({"status": "error", "error": str(exc)})


class UpdateMealArgs(BaseModel):
    meal_id: int = Field(description="Id from get_meals / search_meal_history.")
    items: list[ItemInput] | None = Field(None, description="The FULL corrected item list (replaces the old one).")
    meal_type: Literal["breakfast", "lunch", "dinner", "snack"] | None = None
    confirmed: bool = Field(False, description="True only after the user confirmed an unusually large amount.")


@tool(args_schema=UpdateMealArgs)
def update_meal(
    meal_id: int,
    items: list[ItemInput] | None = None,
    meal_type: str | None = None,
    confirmed: bool = False,
    config: RunnableConfig = None,
) -> str:
    """Correct an existing meal in place ("actually that was 3 rotis not 2"). Pass the full corrected item
    list; the server recomputes totals and the description. Never use log_meal for corrections."""
    try:
        return _json(svc.update_meal(_user(config), meal_id, items, meal_type, confirmed))
    except Exception as exc:
        return _json({"status": "error", "error": str(exc)})


@tool
def delete_meal(meal_id: int, config: RunnableConfig = None) -> str:
    """Delete (soft-delete) a logged meal by id. Returns the updated day totals."""
    try:
        return _json(svc.delete_meal(_user(config), meal_id))
    except Exception as exc:
        return _json({"status": "error", "error": str(exc)})


@tool
def get_meals(
    date: str | None = None, last_n: int | None = None, meal_type: str | None = None, config: RunnableConfig = None
) -> str:
    """List meals with their ids: for a date ('today', 'yesterday', YYYY-MM-DD), or the last N meals.
    Use before corrections and to resolve 'same as yesterday'."""
    try:
        day = svc.parse_day(date) if not last_n else None
        with get_db() as conn:
            meals = db_get_meals(
                conn, _user(config), target_date=day, last_n=last_n or None, meal_type=meal_type or None
            )
        return _json({"date": day, "meals": [_compact_meal(m) for m in meals]})
    except Exception as exc:
        return _json({"status": "error", "error": str(exc)})


@tool
def search_meal_history(query: str, days: int = 30, config: RunnableConfig = None) -> str:
    """Fuzzy-search the user's own past meals by description ("that paneer wrap", "the biryani last week")."""
    try:
        hits = svc.search_history(_user(config), query, days)
        return _json({"matches": [{**_compact_meal(h), "score": h["score"]} for h in hits]})
    except Exception as exc:
        return _json({"status": "error", "error": str(exc)})


@tool
def get_daily_totals(date: str | None = None, config: RunnableConfig = None) -> str:
    """Calories and macros for a day ('today' default, 'yesterday', YYYY-MM-DD) plus goals and what's remaining."""
    try:
        with get_db() as conn:
            return _json(svc.day_summary(conn, _user(config), svc.parse_day(date)))
    except Exception as exc:
        return _json({"status": "error", "error": str(exc)})


# ── memory ───────────────────────────────────────────────────────────────────


@tool
def get_user_memory(query: str | None = None, config: RunnableConfig = None) -> str:
    """Recall stored facts about the user (preferences, allergies, goals, shortcuts like 'my usual')."""
    try:
        with get_db() as conn:
            facts = (
                get_relevant_memories(conn, _user(config), query, top_k=10, threshold=0.3)
                if query
                else get_all_memory(conn, _user(config))
            )
        return _json({"facts": [{k: f[k] for k in ("key", "value", "category")} for f in facts]})
    except Exception as exc:
        return _json({"status": "error", "error": str(exc)})


class SetMemoryArgs(BaseModel):
    key: str = Field(description="Short snake_case key; reuse an existing key to update it.")
    value: str = Field(description="The fact, e.g. 'vegetarian', '140 g protein daily', '2 parathas and chai'.")
    category: Literal["preference", "shortcut", "target", "personal"]


@tool(args_schema=SetMemoryArgs)
def set_user_memory(key: str, value: str, category: str, config: RunnableConfig = None) -> str:
    """Remember a lasting fact for future sessions (diet, allergy, goal, routine meal, personal context)."""
    if category not in CATEGORIES:
        return _json({"status": "error", "error": f"category must be one of {CATEGORIES}"})
    key = normalize_key(key)
    try:
        emb = generate_embedding(f"{key.replace('_', ' ')}: {value}")
        with get_db() as conn:
            upsert_memory(conn, _user(config), key, value.strip(), category, embedding=emb)
        return _json({"status": "remembered", "key": key, "value": value, "category": category})
    except Exception as exc:
        return _json({"status": "error", "error": str(exc)})


@tool
def delete_user_memory(key: str, config: RunnableConfig = None) -> str:
    """Forget a stored fact (e.g. "I'm not vegetarian anymore")."""
    try:
        with get_db() as conn:
            ok = delete_memory(conn, _user(config), normalize_key(key))
        return _json({"status": "forgotten" if ok else "not_found", "key": key})
    except Exception as exc:
        return _json({"status": "error", "error": str(exc)})


ALL_TOOLS = [
    lookup_nutrition,
    log_meal,
    update_meal,
    delete_meal,
    get_meals,
    search_meal_history,
    get_daily_totals,
    get_user_memory,
    set_user_memory,
    delete_user_memory,
]
