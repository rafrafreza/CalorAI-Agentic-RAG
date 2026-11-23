"""
SQLite persistence: schema, connections and idempotent migrations.

Design decision (kept from v1, and still the right call): daily totals are computed
on the fly with SUM queries instead of a materialised totals table. One source of
truth means a correction can never leave totals stale - the exact bug class
("totals that break on a correction") this product has to rule out.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL DEFAULT 'default',
    timestamp   DATETIME NOT NULL DEFAULT (datetime('now', 'localtime')),
    meal_type   TEXT,
    description TEXT NOT NULL,
    items       TEXT NOT NULL DEFAULT '[]',
    calories    REAL NOT NULL DEFAULT 0,
    protein_g   REAL NOT NULL DEFAULT 0,
    carbs_g     REAL NOT NULL DEFAULT 0,
    fat_g       REAL NOT NULL DEFAULT 0,
    fiber_g     REAL NOT NULL DEFAULT 0,
    source      TEXT NOT NULL DEFAULT 'text',
    raw_input   TEXT,
    image_path  TEXT,
    is_deleted  INTEGER NOT NULL DEFAULT 0,
    created_at  DATETIME NOT NULL DEFAULT (datetime('now')),
    updated_at  DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL DEFAULT 'default',
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    category    TEXT NOT NULL,
    embedding   TEXT,
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_at  DATETIME NOT NULL DEFAULT (datetime('now')),
    updated_at  DATETIME NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, key)
);

CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL DEFAULT 'default',
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    meta        TEXT,
    timestamp   DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_meals_user_date
    ON meals(user_id, timestamp) WHERE is_deleted = 0;
CREATE INDEX IF NOT EXISTS idx_memory_user
    ON memory(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session
    ON conversations(user_id, session_id);
"""

# (table, column, DDL) - additive migrations so v1 databases keep working.
_MIGRATIONS: list[tuple[str, str, str]] = [
    ("conversations", "meta", "ALTER TABLE conversations ADD COLUMN meta TEXT"),
]


def db_path(override: str | None = None) -> str:
    """Resolved at call time so tests (and operators) can point at another file."""
    return override or os.getenv("CALORAI_DB_PATH", "calorai.db")


def get_connection(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(path), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def get_db(path: str | None = None):
    """Connection context manager with commit-on-success / rollback-on-error."""
    conn = get_connection(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: str | None = None) -> None:
    """Create tables and apply additive migrations. Safe to call repeatedly."""
    with get_db(path) as conn:
        conn.executescript(SCHEMA_SQL)
        for table, column, ddl in _MIGRATIONS:
            cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                conn.execute(ddl)


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {db_path()}")
