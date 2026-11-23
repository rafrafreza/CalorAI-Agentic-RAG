"""
NutritionRAG: the facade the agent, the tools and the API talk to.

    ground("biryani", 0.667, "box", profile)  -> Grounding(status, match, nutrition, citation, ...)
    ground_message("had 2 parathas and chai")  -> [Grounding, Grounding]   (speculative retrieval)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from backend.rag.corpus import Corpus, FoodDoc, default_corpus
from backend.rag.dense import DenseIndex
from backend.rag.mentions import Mention, extract_mentions, protected_phrases
from backend.rag.portion import ScaledNutrition, scale
from backend.rag.profile import DietProfile
from backend.rag.retriever import Candidate, Decision, HybridRetriever, SearchResult

log = logging.getLogger("calorai.rag")

MATERIALITY = 0.20  # an alternative is "material" if it changes the calorie estimate by >20%


@dataclass
class Grounding:
    query: str
    quantity: float | None
    unit: str | None
    vague: bool
    status: Decision
    confidence: float
    match: FoodDoc | None = None
    nutrition: ScaledNutrition | None = None
    alternatives: list[dict] = field(default_factory=list)
    material_ambiguity: bool = False
    conflicts: list[str] = field(default_factory=list)
    profile_promoted: bool = False
    escalated: bool = False
    channels: tuple[str, ...] = ()
    latency_ms: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Safe to log without asking: grounded, or ambiguous but not materially so."""
        return self.match is not None and (
            self.status == "grounded" or (self.status == "ambiguous" and not self.material_ambiguity)
        )

    def to_llm(self) -> dict:
        """Compact payload for the model's context window."""
        base = {"food": self.query, "status": self.status, "confidence": round(self.confidence, 2)}
        if self.quantity is not None:
            base["quantity"] = self.quantity
        if self.unit:
            base["unit"] = self.unit
        if self.vague:
            base["quantity_is_vague"] = True
        if self.status == "no_match" or not self.match:
            base["note"] = "Not in the knowledge base. Estimate per-item macros yourself and label them as estimates."
            if self.candidates:
                base["closest"] = [c.doc.name for c in self.candidates[:2]]
            return base
        n = self.nutrition
        base.update(
            {
                "kb_id": self.match.id,
                "name": self.match.name,
                "serving": self.match.serving_label,
                "nutrition_for_quantity": {k: v for k, v in n.as_dict().items()} if n else None,
                "source": self.match.source,
            }
        )
        if n and n.assumptions:
            base["assumptions"] = n.assumptions
        if self.alternatives:
            base["alternatives"] = self.alternatives
        if self.status == "ambiguous":
            base["material_ambiguity"] = self.material_ambiguity
        if self.conflicts:
            base["conflicts"] = self.conflicts
        if self.profile_promoted:
            base["note"] = "Chose the variant that fits the user's dietary profile."
        return base

    def to_ui(self) -> dict:
        """Rich payload for the UI's retrieval inspector."""
        payload = self.to_llm()
        payload.update(
            {
                "match_name": self.match.name if self.match else None,
                "citation": self.match.citation() if self.match else None,
                "diet": self.match.diet if self.match else None,
                "allergens": list(self.match.allergens) if self.match else [],
                "channels": list(self.channels),
                "escalated": self.escalated,
                "latency_ms": self.latency_ms,
                "candidates": [c.explain() for c in self.candidates[:5]],
            }
        )
        return payload


class NutritionRAG:
    def __init__(self, corpus: Corpus, dense: DenseIndex | None = None):
        self.corpus = corpus
        self.dense = dense
        self.retriever = HybridRetriever(corpus, dense=dense)
        self.protected = protected_phrases(s for d in corpus.docs for s in d.surface_forms)

    # ------------------------------------------------------------------ queries
    def get(self, food_id: str | None) -> FoodDoc | None:
        return self.corpus.get(food_id)

    def search(self, query: str, k: int = 5, profile: DietProfile | None = None, **kw) -> SearchResult:
        return self.retriever.search(query, k=k, profile=profile, **kw)

    def ground(
        self,
        food: str,
        quantity: float | None = None,
        unit: str | None = None,
        profile: DietProfile | None = None,
        vague: bool = False,
    ) -> Grounding:
        res = self.retriever.search(food, k=5, profile=profile)
        g = Grounding(
            query=food,
            quantity=quantity,
            unit=unit,
            vague=vague,
            status=res.decision,
            confidence=res.confidence,
            profile_promoted=res.profile_promoted,
            escalated=res.escalated,
            channels=res.channels_used,
            latency_ms=res.latency_ms,
            candidates=res.candidates,
        )
        top = res.top
        if top is None or res.decision == "no_match":
            return g
        g.match = top.doc
        g.conflicts = list(top.conflicts)
        try:
            g.nutrition = scale(top.doc, quantity, unit)
        except ValueError:
            g.nutrition = scale(top.doc, None, None)
        margin = self.retriever.config.ambiguity_margin
        rivals = [c for c in res.candidates[1:4] if c.score >= top.score - max(margin, 0.12)]
        for c in rivals:
            kcal = scale(c.doc, quantity, unit).kcal
            g.alternatives.append({"kb_id": c.doc.id, "name": c.doc.name, "kcal": kcal})
            # Variants the user cannot eat are not plausible readings of what they ate.
            if c.conflicts:
                continue
            close = res.decision == "ambiguous" and c.score >= top.score - margin
            if close and g.nutrition.kcal and abs(kcal - g.nutrition.kcal) / g.nutrition.kcal > MATERIALITY:
                g.material_ambiguity = True
        return g

    def ground_mentions(self, mentions: list[Mention], profile: DietProfile | None = None) -> list[Grounding]:
        if self.dense is not None and self.dense.ready:
            self.dense.warm_queries([m.food for m in mentions])  # one embedding round trip per turn
        return [self.ground(m.food, m.quantity, m.unit, profile, vague=m.vague) for m in mentions]

    def ground_message(self, message: str, profile: DietProfile | None = None) -> list[Grounding]:
        return self.ground_mentions(extract_mentions(message, self.protected), profile)

    def extract(self, message: str) -> list[Mention]:
        return extract_mentions(message, self.protected)

    def stats(self) -> dict:
        dense_status = self.dense.status if self.dense else "disabled"
        return {
            **self.corpus.stats(),
            "channels": {"bm25": "ready", "fuzzy": "ready", "dense": dense_status},
            "config": {k: v for k, v in vars(self.retriever.config).items() if not k.startswith("_")},
        }


_lock = threading.Lock()
_instance: NutritionRAG | None = None


def init_rag(
    api_key: str | None = None,
    *,
    enable_dense: bool = False,
    embedding_model: str = "gemini-embedding-001",
    embedding_dims: int = 768,
    cache_dir=".cache/dense",
    background: bool = True,
) -> NutritionRAG:
    """Build the singleton. The dense index (if enabled) builds in the background."""
    global _instance
    with _lock:
        corpus = default_corpus()
        dense = None
        if enable_dense and api_key:
            try:
                from backend.rag.dense import GeminiEmbedder

                dense = DenseIndex(corpus, GeminiEmbedder(api_key, embedding_model, embedding_dims), cache_dir)
                dense.build_in_background() if background else dense.build()
            except Exception as exc:  # never block startup on the optional channel
                log.warning("dense channel disabled: %s", exc)
                dense = None
        _instance = NutritionRAG(corpus, dense)
        return _instance


def get_rag() -> NutritionRAG:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = NutritionRAG(default_corpus())
    return _instance
