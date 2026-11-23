"""
Vision pipeline: photo -> structured food items (a separate model call from the conversation).

The vision model ONLY identifies foods and portions. It never converses, never calls
tools and never produces calorie numbers: its items are grounded through the same RAG
engine as typed text, so a photo of biryani and the word "biryani" get the same numbers.

Latency: images are downscaled to <= 1024 px JPEG before upload (phone photos are
often 4-12 MP; the model doesn't need that) and the response is schema-constrained
JSON, so there is no fragile free-text parsing.
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import BaseModel

from backend.agent.prompts import VISION_PROMPT
from backend.core.config import get_settings
from backend.models.schemas import MealItem, VisionResult

MAX_SIDE = 1024


class _SchemaItem(BaseModel):
    name: str
    quantity: float
    unit: str


class _VisionSchema(BaseModel):
    """Constraint-free mirror of VisionResult for Gemini's response_schema (no defaults/bounds)."""

    description: str
    items: list[_SchemaItem]
    confidence: float
    notes: str


def prepare_image(path: str | Path, max_side: int = MAX_SIDE) -> bytes:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()


@lru_cache(maxsize=1)
def _client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def extract_food_from_image(image_path: str) -> VisionResult:
    """Blocking call (run it in a worker thread). Raises if the model is unavailable."""
    from google.genai import types

    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("Photo analysis requires GOOGLE_API_KEY")
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    response = _client(settings.google_api_key).models.generate_content(
        model=settings.vision_model,
        contents=[VISION_PROMPT, types.Part.from_bytes(data=prepare_image(path), mime_type="image/jpeg")],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=_VisionSchema,
        ),
    )
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, _VisionSchema):
        return VisionResult(
            description=parsed.description,
            items=[MealItem(**i.model_dump()) for i in parsed.items],
            confidence=min(1.0, max(0.0, parsed.confidence)),
            notes=parsed.notes,
        )
    raw = (response.text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return VisionResult.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        return VisionResult(
            description=raw[:200] or "Unrecognised image",
            items=[],
            confidence=0.3,
            notes="Vision model returned an unexpected format.",
        )


def summarize_for_agent(result: VisionResult, caption: str) -> str:
    """Merge the vision output and the user's caption into ONE user turn (photo + caption = one meal)."""
    items = ", ".join(f"{i.quantity:g} {i.unit} {i.name}" for i in result.items) or "no food detected"
    lines = [
        f"[Photo analysis — confidence {result.confidence:.0%}]",
        f"Detected: {result.description}",
        f"Items: {items}",
    ]
    if result.notes:
        lines.append(f"Notes: {result.notes}")
    if result.confidence < 0.7:
        lines.append("(Low confidence: confirm the items with the user before logging.)")
    lines.append(f"User caption: {caption.strip()}" if caption and caption.strip() else "User caption: (none)")
    return "\n".join(lines)


__all__ = ["MealItem", "VisionResult", "extract_food_from_image", "prepare_image", "summarize_for_agent"]
