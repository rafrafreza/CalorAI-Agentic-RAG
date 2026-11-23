"""
Memory WRITE path - runs after the response has been sent (never adds user-facing latency).

1. A cheap lexical gate skips turns that cannot contain durable facts (most turns are
   plain food logs), saving an LLM call per message.
2. Structured output (Pydantic schema) instead of parsing free-form JSON.
3. Existing keys are passed in so updates reuse keys instead of creating near-duplicates.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from backend.agent.prompts import MEMORY_EXTRACTION_PROMPT
from backend.core.llm import get_chat_model
from backend.db.database import get_db
from backend.db.memory import generate_embedding, get_all_memory, normalize_key, upsert_memory

log = logging.getLogger("calorai.memory")

_FACT_CUES = re.compile(
    r"\b(i'?m|i am|i don'?t|i do not|i never|i always|i usually|my usual|usual|allerg|intoleran|vegetarian|vegan|"
    r"pescatarian|eggetarian|keto|diet|goal|target|aim|trying to|want to|prefer|love|hate|can'?t eat|avoid|"
    r"brother|sister|wife|husband|partner|mom|dad|kid|roommate|work|shift|gym|train|remember|from now on)\b",
    re.I,
)


class Fact(BaseModel):
    key: str = Field(description="short snake_case key")
    value: str = Field(description="the fact")
    category: Literal["preference", "shortcut", "target", "personal"]


class Facts(BaseModel):
    facts: list[Fact] = Field(default_factory=list)


def worth_extracting(user_message: str) -> bool:
    return bool(_FACT_CUES.search(user_message or ""))


async def extract_and_store(user_id: str, user_message: str, assistant_message: str) -> list[dict]:
    if not worth_extracting(user_message):
        return []
    llm = get_chat_model("memory")
    if llm is None:
        return []
    with get_db() as conn:
        existing = [m["key"] for m in get_all_memory(conn, user_id)]
    prompt = MEMORY_EXTRACTION_PROMPT.format(
        existing_keys=", ".join(existing) or "(none)",
        user_message=user_message,
        assistant_message=assistant_message,
    )
    try:
        result: Facts = await llm.with_structured_output(Facts).ainvoke(prompt)
    except Exception as exc:  # best effort: memory must never break a turn
        log.warning("memory extraction failed: %s", exc)
        return []
    stored = []
    for fact in (result.facts if result else [])[:5]:
        key, value = normalize_key(fact.key), fact.value.strip()
        if not value:
            continue
        emb = await asyncio.to_thread(generate_embedding, f"{key.replace('_', ' ')}: {value}")
        with get_db() as conn:
            upsert_memory(conn, user_id, key, value, fact.category, embedding=emb)
        stored.append({"key": key, "value": value, "category": fact.category})
    if stored:
        log.info("stored %d memory fact(s) for %s", len(stored), user_id)
    return stored
