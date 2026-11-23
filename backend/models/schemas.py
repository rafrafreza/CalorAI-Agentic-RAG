"""Pydantic data contracts shared by the API, the agent and the vision pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MealItem(BaseModel):
    """A food item detected by the vision model."""

    name: str
    quantity: float = 1.0
    unit: str = "serving"


class VisionResult(BaseModel):
    """Structured output of the vision model."""

    description: str = Field(description="Short description of the whole plate")
    items: list[MealItem] = Field(default_factory=list)
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    notes: str = ""


class ChatResponse(BaseModel):
    """Non-streaming chat response (the streaming endpoint emits the same data as SSE events)."""

    response: str
    session_id: str
    mode: str
    latency_ms: float
    meals_changed: list[dict] = Field(default_factory=list)
    groundings: list[dict] = Field(default_factory=list)
    memories_used: list[dict] = Field(default_factory=list)
    day: dict | None = None
    trace: list[dict] = Field(default_factory=list)
