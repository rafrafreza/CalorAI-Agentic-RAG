"""LangGraph state for one chat turn."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    session_id: str
    user_text: str  # the raw message (before any vision summary is merged in)
    image_path: str | None
    vision: dict | None  # structured output of the vision model
    memory_context: str  # rendered profile block for the system prompt
    memories_used: list[dict]  # pinned + retrieved facts (for the UI)
    profile: dict  # {"diet": ..., "allergens": [...]}
    groundings: list[dict]  # speculative retrieval results (UI payloads)
    grounding_context: str  # rendered retrieval block for the system prompt
    meal_type_hint: str | None
    actions: Annotated[list[dict], operator.add]  # tool / offline-engine results, in order
    trace: Annotated[list[dict], operator.add]  # timed spans for every node
