"""
Hybrid retriever: multi-channel candidate generation -> Reciprocal Rank Fusion ->
feature reranker -> calibrated decision -> profile-aware disambiguation.

Design notes
------------
* **Channels.** BM25 catches exact/lexical evidence and English glosses ("lentil curry");
  the phonetic char-n-gram channel catches typos and transliterations ("parontha",
  "biriyani"); the optional dense channel catches paraphrase ("fried milk balls in syrup").
* **Fusion.** RRF needs no score calibration across channels, which is exactly the
  problem with mixing BM25 scores, cosine similarities and embedding distances.
* **Reranking.** A small set of interpretable features (exact surface-form match,
  soft token coverage, best-form F1, fuzzy similarity, fused rank, prior). Every
  candidate carries its features so the UI and evals can explain *why* it won.
* **Decision.** The top score is compared to calibrated thresholds: confident
  matches are ``grounded``, near-ties are ``ambiguous`` (with a materiality check
  done by the service), weak matches are ``no_match`` -> the agent must estimate
  and say so instead of pretending a wrong food is a match.
* **Adaptive dense escalation.** The embedding call costs a network round trip, so
  it only runs when the cheap lexical channels are not confident.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from backend.rag.corpus import Corpus, FoodDoc
from backend.rag.indexes import BM25Index, CharNgramIndex
from backend.rag.profile import PREFERRED_DIETS, DietProfile
from backend.rag.text import canonical_key, term_views, terms, tokens, view_similarity

if TYPE_CHECKING:
    from backend.rag.dense import DenseIndex

Decision = Literal["grounded", "ambiguous", "no_match"]
_PERCENT = re.compile(r"^\s*\d+(?:\.\d+)?\s*%")
ALL_CHANNELS: tuple[str, ...] = ("bm25", "fuzzy", "dense")


@dataclass(frozen=True)
class RetrieverConfig:
    pool: int = 25
    rrf_k: int = 60
    channel_weights: tuple[tuple[str, float], ...] = (("bm25", 1.0), ("fuzzy", 1.0), ("dense", 1.0))
    # reranker feature weights (normalised by the sum of *active* weights)
    w_exact: float = 0.30
    w_form_f1: float = 0.20
    w_coverage: float = 0.15
    w_fuzzy: float = 0.20
    w_fused: float = 0.10
    w_prior: float = 0.02
    w_dense: float = 0.15
    # Queries whose words are not all explained by a candidate's names are demoted:
    # "dragon fruit smoothie" must not ground to "banana smoothie".
    min_coverage: float = 0.7
    coverage_penalty: float = 0.75
    # decision calibration
    tau_abstain: float = 0.42
    ambiguity_margin: float = 0.04
    escalate_below: float = 0.62
    profile_margin: float = 0.15

    def weight(self, channel: str) -> float:
        return dict(self.channel_weights).get(channel, 1.0)


@dataclass
class ChannelHit:
    rank: int
    score: float


@dataclass
class Candidate:
    doc: FoodDoc
    score: float = 0.0
    fused: float = 0.0
    channels: dict[str, ChannelHit] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)

    def explain(self) -> dict:
        return {
            "id": self.doc.id,
            "name": self.doc.name,
            "score": round(self.score, 4),
            "fused": round(self.fused, 5),
            "channels": {c: {"rank": h.rank, "score": round(h.score, 4)} for c, h in self.channels.items()},
            "features": {k: round(v, 3) for k, v in self.features.items()},
            "conflicts": self.conflicts,
        }


@dataclass
class SearchResult:
    query: str
    candidates: list[Candidate]
    decision: Decision
    confidence: float
    channels_used: tuple[str, ...]
    escalated: bool = False
    profile_promoted: bool = False
    latency_ms: float = 0.0

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


class HybridRetriever:
    def __init__(self, corpus: Corpus, config: RetrieverConfig | None = None, dense: DenseIndex | None = None):
        self.corpus = corpus
        self.config = config or RetrieverConfig()
        self.dense = dense
        docs = corpus.docs
        self.bm25 = BM25Index([d.search_text for d in docs])
        self.fuzzy = CharNgramIndex((d.index, s) for d in docs for s in d.surface_forms)
        self.form_terms: list[list[tuple[tuple[str, str], ...]]] = [
            [tuple(term_views(s)) for s in d.surface_forms] for d in docs
        ]
        self.form_keys: list[frozenset[str]] = [frozenset(canonical_key(s) for s in d.surface_forms) for d in docs]

    # ------------------------------------------------------------------ public
    def with_config(self, **overrides) -> HybridRetriever:
        """Cheap clone sharing the indexes (used by evals for ablations)."""
        clone = object.__new__(HybridRetriever)
        clone.__dict__.update(self.__dict__)
        clone.config = replace(self.config, **overrides)
        return clone

    @property
    def dense_ready(self) -> bool:
        return bool(self.dense and self.dense.ready)

    def search(
        self,
        query: str,
        k: int = 5,
        profile: DietProfile | None = None,
        channels: tuple[str, ...] = ("bm25", "fuzzy", "dense"),
        rerank: bool = True,
        allow_dense: bool = True,
        query_vector=None,
    ) -> SearchResult:
        start = time.perf_counter()
        q_terms = terms(query)
        if not q_terms:
            return SearchResult(query, [], "no_match", 0.0, (), latency_ms=0.0)

        lexical = tuple(c for c in channels if c in ("bm25", "fuzzy"))
        q_views = term_views(query)
        result = self._run(query, q_terms, q_views, lexical, rerank, profile, k)

        wants_dense = "dense" in channels and allow_dense and self.dense_ready
        if wants_dense and (not lexical or result.confidence < self.config.escalate_below):
            vec = query_vector if query_vector is not None else self.dense.embed_query(query)
            if vec is not None:
                result = self._run(query, q_terms, q_views, (*lexical, "dense"), rerank, profile, k, dense_vec=vec)
                result.escalated = True

        result.latency_ms = round((time.perf_counter() - start) * 1000, 3)
        return result

    # ----------------------------------------------------------------- internals
    def _run(self, query, q_terms, q_views, channels, rerank, profile, k, dense_vec=None) -> SearchResult:
        cfg = self.config
        hits: dict[str, list[tuple[int, float]]] = {}
        if "bm25" in channels:
            hits["bm25"] = self.bm25.search(query, cfg.pool)
        if "fuzzy" in channels:
            hits["fuzzy"] = self.fuzzy.search(query, cfg.pool)
        if "dense" in channels and dense_vec is not None:
            hits["dense"] = self.dense.search(dense_vec, cfg.pool)

        cands: dict[int, Candidate] = {}
        for channel, ranked in hits.items():
            w = cfg.weight(channel)
            for rank, (doc_idx, score) in enumerate(ranked, start=1):
                cand = cands.get(doc_idx)
                if cand is None:
                    cand = cands[doc_idx] = Candidate(doc=self.corpus.docs[doc_idx])
                cand.channels[channel] = ChannelHit(rank=rank, score=score)
                cand.fused += w / (cfg.rrf_k + rank)

        if not cands:
            return SearchResult(query, [], "no_match", 0.0, tuple(hits))

        max_fused = sum(cfg.weight(c) / (cfg.rrf_k + 1) for c in hits)
        q_key_num = canonical_key(query, keep_numbers=True)
        q_key = canonical_key(query, keep_numbers=False)
        toks = tokens(query)
        leading_qty = bool(toks) and toks[0].isdigit() and not _PERCENT.match(query)

        for cand in cands.values():
            fused_norm = cand.fused / max_fused if max_fused else 0.0
            if not rerank:
                cand.score = fused_norm
                cand.features = {"fused": fused_norm}
                continue
            feats = self._features(cand, q_views, q_key_num, q_key, leading_qty, fused_norm)
            cand.features = feats
            weights = {
                "exact": cfg.w_exact,
                "form_f1": cfg.w_form_f1,
                "coverage": cfg.w_coverage,
                "fuzzy": cfg.w_fuzzy,
                "fused": cfg.w_fused,
                "prior": cfg.w_prior,
            }
            if "dense" in hits:
                weights["dense"] = cfg.w_dense
            total = sum(weights.values())
            cand.score = sum(weights[f] * feats.get(f, 0.0) for f in weights) / total
            if not feats["exact"] and feats["coverage"] < cfg.min_coverage:
                cand.score *= cfg.coverage_penalty

        ranked = sorted(cands.values(), key=lambda c: (-c.score, -c.fused, c.doc.index))
        if profile and not profile.is_empty:
            for cand in ranked:
                cand.conflicts = profile.conflicts(cand.doc)

        decision, confidence = self._decide(ranked)
        promoted = False
        if profile and not profile.is_empty and ranked and ranked[0].conflicts:
            promoted = self._promote_compatible(ranked, q_terms, profile)
        return SearchResult(query, ranked[: max(k, 1)], decision, confidence, tuple(hits), profile_promoted=promoted)

    def _features(self, cand, q_views, q_key_num, q_key, leading_qty, fused_norm) -> dict[str, float]:
        idx = cand.doc.index
        keys = self.form_keys[idx]
        exact = 1.0 if (q_key_num in keys or (leading_qty and q_key in keys)) else 0.0

        forms = self.form_terms[idx]
        doc_vocab = {t for form in forms for t in form}
        n_q = max(len(q_views), 1)
        coverage = sum(max((view_similarity(q, d) for d in doc_vocab), default=0.0) for q in q_views) / n_q

        best_f1 = 0.0
        for form in forms:
            if not form:
                continue
            rec = sum(max((view_similarity(q, f) for f in form), default=0.0) for q in q_views) / n_q
            prec = sum(max((view_similarity(f, q) for q in q_views), default=0.0) for f in form) / len(form)
            if rec + prec:
                best_f1 = max(best_f1, 2 * rec * prec / (rec + prec))

        feats = {
            "exact": exact,
            "form_f1": best_f1,
            "coverage": coverage,
            "fuzzy": cand.channels["fuzzy"].score if "fuzzy" in cand.channels else 0.0,
            "fused": fused_norm,
            "prior": cand.doc.prior,
        }
        if "dense" in cand.channels:
            feats["dense"] = max(0.0, cand.channels["dense"].score)
        return feats

    def _decide(self, ranked: list[Candidate]) -> tuple[Decision, float]:
        cfg = self.config
        top = ranked[0].score
        second = ranked[1].score if len(ranked) > 1 else 0.0
        if top < cfg.tau_abstain:
            return "no_match", round(top, 4)
        if top - second < cfg.ambiguity_margin:
            return "ambiguous", round(top, 4)
        return "grounded", round(top, 4)

    def _promote_compatible(self, ranked: list[Candidate], q_terms: list[str], profile: DietProfile) -> bool:
        """
        If the best match conflicts with the user's diet/allergies and the user did
        not explicitly ask for it, prefer the best compatible candidate within a margin.
        ("biryani" for a vegetarian -> Veg Biryani; "chicken biryani" stays chicken.)
        """
        top = ranked[0]
        if profile.is_explicit_choice(q_terms, top.doc):
            return False
        floor = top.score - self.config.profile_margin
        compatible = [(i, c) for i, c in enumerate(ranked[1:], start=1) if c.score >= floor and not c.conflicts]
        if not compatible:
            return False
        # Prefer the closest diet (a vegetarian gets Veg Biryani before Egg Biryani), then score.
        preferred = PREFERRED_DIETS.get(profile.diet or "", frozenset())
        best_i, _ = min(compatible, key=lambda ic: (ic[1].doc.diet not in preferred, -ic[1].score))
        ranked.insert(0, ranked.pop(best_i))
        return True
