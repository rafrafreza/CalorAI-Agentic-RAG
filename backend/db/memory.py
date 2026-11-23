"""
Long-term memory and conversation history.

Memory is NOT conversation history. Memory holds durable facts:

    preference  "vegetarian", "allergic to peanuts"      -> ALWAYS injected (safety)
    target      "140 g protein a day"                     -> ALWAYS injected
    shortcut    "my usual breakfast = 2 parathas + chai"  -> retrieved when relevant
    personal    "brother often shares my meals"           -> retrieved when relevant

Why the split: v1 retrieved *every* memory by embedding similarity with a 0.5
threshold, which means an allergy could silently drop out of context whenever the
message wasn't semantically close to it. Safety-critical facts are now pinned;
only the long tail competes for retrieval (hybrid lexical + optional dense).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re

from backend.rag.text import term_views, view_similarity

log = logging.getLogger("calorai.memory")

CATEGORIES = ("preference", "shortcut", "target", "personal")
PINNED_CATEGORIES = ("preference", "target")
_REFERENCE_CUES = re.compile(
    r"\b(usual|regular|same|always|again|like yesterday|the one|my go[- ]to|as before)\b", re.I
)


# ── Embeddings (optional) ────────────────────────────────────────────────────


def generate_embedding(text: str) -> list[float] | None:
    """Embed a memory fact for storage. Returns None when no embedding backend is configured."""
    from backend.rag.service import get_rag

    dense = get_rag().dense
    if dense is None:
        return None
    try:
        return [float(x) for x in dense.embedder.embed_documents([text])[0]]
    except Exception as exc:
        log.warning("memory embedding failed: %s", exc)
        return None


def embed_query(text: str) -> list[float] | None:
    from backend.rag.service import get_rag

    dense = get_rag().dense
    if dense is None or not dense.ready:
        return None
    vec = dense.embed_query(text)
    return [float(x) for x in vec] if vec is not None else None


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=True))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0


# ── Persistent memory ────────────────────────────────────────────────────────


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")[:64] or "fact"


def upsert_memory(
    conn,
    user_id: str,
    key: str,
    value: str,
    category: str,
    confidence: float = 1.0,
    embedding: list[float] | None = None,
) -> None:
    """Insert or update a fact (unique per user + key)."""
    conn.execute(
        """
        INSERT INTO memory (user_id, key, value, category, confidence, embedding, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(user_id, key) DO UPDATE SET
            value = excluded.value, category = excluded.category, confidence = excluded.confidence,
            embedding = excluded.embedding, updated_at = datetime('now')
        """,
        (user_id, key, value, category, confidence, json.dumps(embedding) if embedding else None),
    )
    conn.commit()


def get_all_memory(conn, user_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT key, value, category, confidence, updated_at FROM memory WHERE user_id = ? ORDER BY category, key",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_memory_by_key(conn, user_id: str, key: str) -> dict | None:
    row = conn.execute(
        "SELECT key, value, category, confidence FROM memory WHERE user_id = ? AND key = ?", (user_id, key)
    ).fetchone()
    return dict(row) if row else None


def delete_memory(conn, user_id: str, key: str) -> bool:
    cursor = conn.execute("DELETE FROM memory WHERE user_id = ? AND key = ?", (user_id, key))
    conn.commit()
    return cursor.rowcount > 0


def _lexical_score(query_views, fact: dict) -> float:
    fact_views = term_views(f"{fact['key'].replace('_', ' ')} {fact['value']}")
    if not query_views or not fact_views:
        return 0.0
    hits = sum(max(view_similarity(q, f) for f in fact_views) for q in query_views)
    return hits / len(query_views)


def get_relevant_memories(conn, user_id: str, user_query: str, top_k: int = 5, threshold: float = 0.5) -> list[dict]:
    """
    Memory retrieve path.

    1. Pinned: every ``preference`` and ``target`` fact (allergies must never be missed).
    2. Retrieved: other facts scored by max(lexical coverage, dense cosine), keeping
       those above ``threshold``; reference cues ("my usual", "same as") pull in shortcuts.
    """
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT key, value, category, embedding FROM memory WHERE user_id = ?", (user_id,)
        ).fetchall()
    ]
    pinned = [r for r in rows if r["category"] in PINNED_CATEGORIES]
    others = [r for r in rows if r["category"] not in PINNED_CATEGORIES]

    query_views = term_views(user_query)
    cue = bool(_REFERENCE_CUES.search(user_query or ""))
    q_emb: list[float] | None = None
    if any(r.get("embedding") for r in others):
        q_emb = embed_query(user_query)

    scored: list[tuple[float, dict]] = []
    for r in others:
        score = _lexical_score(query_views, r)
        if q_emb and r.get("embedding"):
            with contextlib.suppress(TypeError, json.JSONDecodeError):
                score = max(score, cosine_similarity(q_emb, json.loads(r["embedding"])))
        if cue and r["category"] == "shortcut":
            score = max(score, threshold)
        if score >= threshold:
            r["score"] = round(score, 3)
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])

    result = [{**r, "pinned": True} for r in pinned] + [{**r, "pinned": False} for _, r in scored[:top_k]]
    for r in result:
        r.pop("embedding", None)
    return result


# ── Conversation history ─────────────────────────────────────────────────────


def save_message(conn, user_id: str, session_id: str, role: str, content: str, meta: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO conversations (user_id, session_id, role, content, meta) VALUES (?, ?, ?, ?, ?)",
        (user_id, session_id, role, content, json.dumps(meta) if meta else None),
    )
    conn.commit()


def get_conversation_history(conn, user_id: str, session_id: str, limit: int = 20) -> list[dict]:
    """Most recent ``limit`` messages of a session, oldest first."""
    rows = conn.execute(
        """
        SELECT role, content, meta, timestamp FROM conversations
        WHERE user_id = ? AND session_id = ?
        ORDER BY id DESC LIMIT ?
        """,
        (user_id, session_id, limit),
    ).fetchall()
    out = []
    for r in reversed(rows):
        item = dict(r)
        try:
            item["meta"] = json.loads(item["meta"]) if item.get("meta") else None
        except json.JSONDecodeError:
            item["meta"] = None
        out.append(item)
    return out


def clear_session(conn, user_id: str, session_id: str) -> int:
    cursor = conn.execute("DELETE FROM conversations WHERE user_id = ? AND session_id = ?", (user_id, session_id))
    conn.commit()
    return cursor.rowcount


__all__ = [
    "CATEGORIES",
    "PINNED_CATEGORIES",
    "clear_session",
    "cosine_similarity",
    "delete_memory",
    "embed_query",
    "generate_embedding",
    "get_all_memory",
    "get_conversation_history",
    "get_memory_by_key",
    "get_relevant_memories",
    "normalize_key",
    "save_message",
    "upsert_memory",
]
