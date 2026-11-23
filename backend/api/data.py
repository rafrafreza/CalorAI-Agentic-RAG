"""Read/modify endpoints behind the dashboard: meals, totals, trend, memory, history."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.db import meals as meal_db
from backend.db.database import get_db
from backend.db.memory import clear_session, delete_memory, get_all_memory, get_conversation_history
from backend.rag.profile import profile_from_memories
from backend.services import nutrition_log as svc

router = APIRouter(prefix="/api", tags=["data"])


def _day(date: str | None) -> str:
    try:
        return svc.parse_day(date)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/dashboard")
def dashboard(user_id: str = "default", date: str | None = None):
    """Everything the dashboard renders, in one round trip."""
    day = _day(date)
    with get_db() as conn:
        summary = svc.day_summary(conn, user_id, day)
        meals = [svc.public_meal(m) for m in meal_db.get_meals(conn, user_id, target_date=day)]
        trend = meal_db.get_range_totals(conn, user_id, day, days=7)
        memory = get_all_memory(conn, user_id)
    items = [i for m in meals for i in m["items"]]
    total = sum(float(i.get("calories") or 0) for i in items)
    grounded = sum(float(i.get("calories") or 0) for i in items if i.get("grounded"))
    return {
        "date": day,
        "day": summary,
        "meals": meals,
        "trend": trend,
        "memory": memory,
        "profile": profile_from_memories(memory).as_dict(),
        "grounded_share": round(grounded / total, 3) if total else None,
        "items_logged": len(items),
    }


@router.get("/meals")
def list_meals(user_id: str = "default", date: str | None = None):
    day = _day(date)
    with get_db() as conn:
        return {"date": day, "meals": [svc.public_meal(m) for m in meal_db.get_meals(conn, user_id, target_date=day)]}


@router.delete("/meals/{meal_id}")
def remove_meal(meal_id: int, user_id: str = "default"):
    result = svc.delete_meal(user_id, meal_id)
    if result["status"] != "deleted":
        raise HTTPException(404, result.get("error", "not found"))
    return result


@router.get("/totals")
def totals(user_id: str = "default", date: str | None = None):
    with get_db() as conn:
        return svc.day_summary(conn, user_id, _day(date))


@router.get("/memory")
def memory(user_id: str = "default"):
    with get_db() as conn:
        facts = get_all_memory(conn, user_id)
    return {"memories": facts, "count": len(facts), "profile": profile_from_memories(facts).as_dict()}


@router.delete("/memory/{key}")
def forget(key: str, user_id: str = "default"):
    with get_db() as conn:
        if not delete_memory(conn, user_id, key):
            raise HTTPException(404, f"No memory '{key}'")
    return {"status": "forgotten", "key": key}


@router.get("/history")
def history(session_id: str, user_id: str = "default", limit: int = Query(60, ge=1, le=200)):
    with get_db() as conn:
        return {"session_id": session_id, "messages": get_conversation_history(conn, user_id, session_id, limit)}


@router.delete("/history")
def reset_history(session_id: str, user_id: str = "default"):
    with get_db() as conn:
        return {"deleted": clear_session(conn, user_id, session_id)}
