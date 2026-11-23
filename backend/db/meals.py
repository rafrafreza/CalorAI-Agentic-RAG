"""
Meal persistence. All reads filter ``is_deleted = 0`` (soft delete) and totals are
always re-aggregated from the meals table - never cached.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

MEAL_FIELDS = {"description", "items", "calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "meal_type", "raw_input"}


def _row(row) -> dict:
    meal = dict(row)
    try:
        meal["items"] = json.loads(meal.get("items") or "[]")
    except (TypeError, json.JSONDecodeError):
        meal["items"] = []
    return meal


def create_meal(
    conn,
    user_id: str,
    description: str,
    items: list[dict],
    calories: float,
    protein_g: float,
    carbs_g: float,
    fat_g: float,
    fiber_g: float = 0,
    meal_type: str | None = None,
    source: str = "text",
    raw_input: str | None = None,
    image_path: str | None = None,
    target_date: str | None = None,
) -> int:
    """Insert a meal and return its id. Past-dated meals are stamped at noon local time."""
    now = datetime.now()
    timestamp = (
        f"{target_date} 12:00:00"
        if target_date and target_date != now.date().isoformat()
        else now.isoformat(sep=" ", timespec="seconds")
    )
    cursor = conn.execute(
        """
        INSERT INTO meals (user_id, meal_type, description, items, calories, protein_g, carbs_g,
                           fat_g, fiber_g, source, raw_input, image_path, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            meal_type,
            description,
            json.dumps(items),
            calories,
            protein_g,
            carbs_g,
            fat_g,
            fiber_g,
            source,
            raw_input,
            image_path,
            timestamp,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def update_meal(conn, meal_id: int, user_id: str | None = None, **fields) -> bool:
    """
    Update fields of an existing meal *in place* - the correction path
    ("actually that was 3 rotis not 2"). Returns True only if a live row changed.
    """
    updates = {k: v for k, v in fields.items() if k in MEAL_FIELDS and v is not None}
    if not updates:
        return False
    if isinstance(updates.get("items"), list):
        updates["items"] = json.dumps(updates["items"])
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    sql = f"UPDATE meals SET {set_clause}, updated_at = datetime('now') WHERE id = ? AND is_deleted = 0"
    params: list = [*updates.values(), meal_id]
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)
    cursor = conn.execute(sql, params)
    conn.commit()
    return cursor.rowcount > 0


def delete_meal(conn, meal_id: int, user_id: str | None = None) -> bool:
    """Soft-delete a meal. Returns True only if a live row was deleted."""
    sql = "UPDATE meals SET is_deleted = 1, updated_at = datetime('now') WHERE id = ? AND is_deleted = 0"
    params: list = [meal_id]
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)
    cursor = conn.execute(sql, params)
    conn.commit()
    return cursor.rowcount > 0


def get_meal_by_id(conn, meal_id: int, user_id: str | None = None) -> dict | None:
    sql = "SELECT * FROM meals WHERE id = ? AND is_deleted = 0"
    params: list = [meal_id]
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)
    row = conn.execute(sql, params).fetchone()
    return _row(row) if row else None


def get_meals(
    conn,
    user_id: str,
    target_date: str | None = None,
    last_n: int | None = None,
    meal_type: str | None = None,
) -> list[dict]:
    """Meals for a date (default today), or the N most recent regardless of date."""
    query = "SELECT * FROM meals WHERE user_id = ? AND is_deleted = 0"
    params: list = [user_id]
    if meal_type:
        query += " AND meal_type = ?"
        params.append(meal_type)
    if last_n:
        query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(last_n)
    else:
        query += " AND date(timestamp) = ? ORDER BY timestamp ASC, id ASC"
        params.append(target_date or date.today().isoformat())
    return [_row(r) for r in conn.execute(query, params).fetchall()]


def get_meals_since(conn, user_id: str, since: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM meals WHERE user_id = ? AND is_deleted = 0 AND date(timestamp) >= ? "
        "ORDER BY timestamp DESC, id DESC",
        (user_id, since),
    ).fetchall()
    return [_row(r) for r in rows]


def get_daily_totals(conn, user_id: str, target_date: str | None = None) -> dict:
    """Single source of truth for totals: re-aggregated from meals on every call."""
    target_date = target_date or date.today().isoformat()
    row = conn.execute(
        """
        SELECT COALESCE(SUM(calories), 0)  AS total_calories,
               COALESCE(SUM(protein_g), 0) AS total_protein_g,
               COALESCE(SUM(carbs_g), 0)   AS total_carbs_g,
               COALESCE(SUM(fat_g), 0)     AS total_fat_g,
               COALESCE(SUM(fiber_g), 0)   AS total_fiber_g,
               COUNT(*)                    AS meal_count
        FROM meals
        WHERE user_id = ? AND date(timestamp) = ? AND is_deleted = 0
        """,
        (user_id, target_date),
    ).fetchone()
    return {
        "date": target_date,
        "total_calories": round(row["total_calories"], 1),
        "total_protein_g": round(row["total_protein_g"], 1),
        "total_carbs_g": round(row["total_carbs_g"], 1),
        "total_fat_g": round(row["total_fat_g"], 1),
        "total_fiber_g": round(row["total_fiber_g"], 1),
        "meal_count": row["meal_count"],
    }


def get_range_totals(conn, user_id: str, end_date: str, days: int = 7) -> list[dict]:
    """Per-day totals for the ``days`` days ending at ``end_date`` (zero-filled)."""
    end = date.fromisoformat(end_date)
    start = end - timedelta(days=days - 1)
    rows = conn.execute(
        """
        SELECT date(timestamp) AS day, SUM(calories) AS calories, SUM(protein_g) AS protein_g,
               SUM(carbs_g) AS carbs_g, SUM(fat_g) AS fat_g, COUNT(*) AS meals
        FROM meals
        WHERE user_id = ? AND is_deleted = 0 AND date(timestamp) BETWEEN ? AND ?
        GROUP BY day
        """,
        (user_id, start.isoformat(), end.isoformat()),
    ).fetchall()
    by_day = {r["day"]: r for r in rows}
    out = []
    for i in range(days):
        d = (start + timedelta(days=i)).isoformat()
        r = by_day.get(d)
        out.append(
            {
                "date": d,
                "calories": round(r["calories"], 1) if r else 0.0,
                "protein_g": round(r["protein_g"], 1) if r else 0.0,
                "carbs_g": round(r["carbs_g"], 1) if r else 0.0,
                "fat_g": round(r["fat_g"], 1) if r else 0.0,
                "meals": r["meals"] if r else 0,
            }
        )
    return out


def find_recent_duplicate(conn, user_id: str, day: str, signature: str, within_seconds: int = 180) -> dict | None:
    """A live meal logged on ``day`` in the last few minutes with the same item signature."""
    rows = conn.execute(
        """
        SELECT * FROM meals
        WHERE user_id = ? AND is_deleted = 0 AND date(timestamp) = ?
          AND created_at >= datetime('now', ?)
        ORDER BY id DESC
        """,
        (user_id, day, f"-{int(within_seconds)} seconds"),
    ).fetchall()
    for r in rows:
        meal = _row(r)
        if item_signature(meal["items"]) == signature:
            return meal
    return None


def item_signature(items: list[dict]) -> str:
    parts = sorted(
        ":".join(
            [
                (i.get("kb_id") or i.get("name") or "").lower(),
                str(round(float(i.get("quantity") or 1), 2)),
                i.get("unit") or "",
            ]
        )
        for i in items
    )
    return "|".join(parts)
