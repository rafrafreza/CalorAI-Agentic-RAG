"""
Retrieval inspection endpoints - the RAG engine is a first-class, explainable component.

    GET /api/rag/search?q=parontha&diet=vegetarian   ranked candidates with per-channel ranks/scores
    GET /api/rag/analyze?text=had 2 parathas and chai mention extraction + grounding of a message
    GET /api/rag/stats                               KB size, channels, calibrated thresholds
    GET /api/rag/foods/{id}                          one KB entry
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.rag.portion import scale
from backend.rag.profile import DIET_CONFLICTS, DietProfile
from backend.rag.service import get_rag

router = APIRouter(prefix="/api/rag", tags=["rag"])


def _profile(diet: str | None, allergens: str | None) -> DietProfile | None:
    if diet and diet not in DIET_CONFLICTS:
        raise HTTPException(422, f"diet must be one of {sorted(DIET_CONFLICTS)}")
    allergy_set = frozenset(a.strip() for a in (allergens or "").split(",") if a.strip())
    return DietProfile(diet=diet or None, allergens=allergy_set) if (diet or allergy_set) else None


def _food(doc) -> dict:
    return {
        "id": doc.id,
        "name": doc.name,
        "aliases": list(doc.aliases),
        "description": doc.description,
        "category": doc.category,
        "cuisine": doc.cuisine,
        "serving": doc.serving_label,
        "per_serving": doc.per_serving(),
        "diet": doc.diet,
        "allergens": list(doc.allergens),
        "source": doc.source,
        "source_label": doc.source_label,
    }


@router.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=200),
    k: int = Query(8, ge=1, le=25),
    diet: str | None = None,
    allergens: str | None = None,
):
    rag = get_rag()
    res = rag.search(q, k=k, profile=_profile(diet, allergens))
    return {
        "query": q,
        "decision": res.decision,
        "confidence": res.confidence,
        "latency_ms": res.latency_ms,
        "channels": list(res.channels_used),
        "escalated": res.escalated,
        "profile_promoted": res.profile_promoted,
        "candidates": [{**c.explain(), "food": _food(c.doc)} for c in res.candidates],
    }


@router.get("/analyze")
def analyze(
    text: str = Query(..., min_length=1, max_length=500), diet: str | None = None, allergens: str | None = None
):
    rag = get_rag()
    mentions = rag.extract(text)
    groundings = rag.ground_mentions(mentions, _profile(diet, allergens))
    total = sum(g.nutrition.kcal for g in groundings if g.usable and g.nutrition)
    return {
        "text": text,
        "mentions": [m.__dict__ for m in mentions],
        "groundings": [g.to_ui() for g in groundings],
        "grounded_kcal": round(total, 1),
    }


@router.get("/stats")
def stats():
    return get_rag().stats()


@router.get("/foods/{food_id}")
def food(food_id: str, quantity: float | None = None, unit: str | None = None):
    doc = get_rag().get(food_id)
    if not doc:
        raise HTTPException(404, f"Unknown food id '{food_id}'")
    out = _food(doc)
    if quantity:
        out["scaled"] = scale(doc, quantity, unit).__dict__
    return out
