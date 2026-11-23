"""Behavioural tests for the hybrid retriever, grounding service and mention extraction."""

import hashlib

import numpy as np
import pytest

from backend.rag.corpus import default_corpus
from backend.rag.dense import DenseIndex
from backend.rag.mentions import detect_meal_type, extract_mentions
from backend.rag.profile import DietProfile, profile_from_memories
from backend.rag.retriever import HybridRetriever
from backend.rag.service import NutritionRAG, get_rag

VEG = DietProfile(diet="vegetarian")
PEANUT = DietProfile(allergens=frozenset({"peanut"}))


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(default_corpus())


def top(retriever, q, profile=None):
    return retriever.search(q, profile=profile)


class TestRetriever:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("paratha", "paratha"),
            ("parontha", "paratha"),
            ("chiken biryani", "chicken_biryani"),
            ("golgappa", "pani_puri"),
            ("murgh makhani", "butter_chicken"),
            ("dhosa", "plain_dosa"),
            ("mac and cheese", "mac_and_cheese"),
            ("chicken 65", "chicken_65"),
            ("2% milk", "toned_milk"),
        ],
    )
    def test_top1(self, retriever, query, expected):
        assert top(retriever, query).top.doc.id == expected

    @pytest.mark.parametrize("query", ["beef wellington", "tiramisu", "how am i doing today", "dragon fruit smoothie"])
    def test_abstains_out_of_knowledge_base(self, retriever, query):
        assert top(retriever, query).decision == "no_match"

    def test_generic_name_is_ambiguous_with_a_sensible_default(self, retriever):
        res = top(retriever, "biryani")
        assert res.decision == "ambiguous"
        assert res.top.doc.id == "chicken_biryani"

    def test_profile_resolves_implicit_choices(self, retriever):
        res = top(retriever, "biryani", VEG)
        assert res.profile_promoted and res.top.doc.id == "veg_biryani"
        assert top(retriever, "tikka masala", VEG).top.doc.id == "paneer_tikka_masala"

    def test_explicit_choices_are_respected_and_flagged(self, retriever):
        res = top(retriever, "chicken biryani", VEG)
        assert res.top.doc.id == "chicken_biryani" and not res.profile_promoted
        assert any("non-veg" in c for c in res.top.conflicts)

    def test_allergen_conflicts_are_surfaced(self, retriever):
        res = top(retriever, "satay", PEANUT)
        assert res.top.doc.id == "chicken_satay"
        assert any("peanut" in c for c in res.top.conflicts)

    def test_candidates_are_explainable(self, retriever):
        explained = top(retriever, "palak paner").top.explain()
        assert {"bm25", "fuzzy"} & set(explained["channels"])
        assert {"exact", "coverage", "form_f1", "fuzzy", "fused"} <= set(explained["features"])

    def test_ablation_clone_shares_indexes(self, retriever):
        clone = retriever.with_config(tau_abstain=0.99)
        assert clone.bm25 is retriever.bm25
        assert clone.search("paratha").decision != "no_match"  # exact match still scores ~1.0
        assert clone.search("parontha").decision == "no_match"


class _HashEmbedder:
    """Deterministic bag-of-words embedder standing in for Gemini in tests."""

    model, dims = "fake-embed", 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dims, dtype=np.float32)
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.strip(".,:").encode()).hexdigest(), 16) % self.dims] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])


class TestDenseChannel:
    def test_escalates_only_when_lexical_is_unsure(self, tmp_path):
        corpus = default_corpus()
        dense = DenseIndex(corpus, _HashEmbedder(), cache_dir=tmp_path)
        dense.build()
        assert dense.ready and dense.cache_path.exists()
        r = HybridRetriever(corpus, dense=dense)
        assert not r.search("paratha").escalated  # confident lexical hit: no embedding call
        res = r.search("fried milk dumplings soaked in sugar syrup")
        assert res.escalated and "dense" in res.channels_used

    def test_failed_build_degrades_gracefully(self, tmp_path):
        class Broken(_HashEmbedder):
            def embed_documents(self, texts):
                raise RuntimeError("quota exceeded")

        dense = DenseIndex(default_corpus(), Broken(), cache_dir=tmp_path)
        dense.build()
        assert dense.status == "failed" and not dense.ready
        r = HybridRetriever(default_corpus(), dense=dense)
        assert r.search("paratha").top.doc.id == "paratha"


class TestGrounding:
    def test_ground_scales_and_cites(self):
        g = get_rag().ground("parathas", 2)
        assert g.status == "grounded" and g.usable
        assert g.nutrition.kcal == 520
        assert g.to_llm()["kb_id"] == "paratha" and g.to_ui()["citation"]["source"] == "recipe-avg"

    def test_materiality(self):
        assert not get_rag().ground("biryani").material_ambiguity  # variants within 20%
        assert get_rag().ground("dumplings").material_ambiguity  # 1 momo vs 5 gyoza

    def test_profile_conflicting_variants_do_not_make_it_material(self):
        g = get_rag().ground("biryani", 0.667, "box", VEG)
        assert g.match.id == "veg_biryani" and g.usable

    def test_no_match_payload_tells_the_model_to_estimate(self):
        payload = get_rag().ground("beef wellington").to_llm()
        assert payload["status"] == "no_match" and "estimate" in payload["note"].lower()

    def test_stats(self):
        stats = NutritionRAG(default_corpus()).stats()
        assert stats["foods"] >= 400 and stats["channels"]["dense"] == "disabled"


class TestMentions:
    protected = get_rag().protected

    def ex(self, text):
        return [(m.quantity, m.unit, m.food) for m in extract_mentions(text, self.protected)]

    def test_split_and_quantities(self):
        assert self.ex("had 2 parathas and chai for breakfast") == [(2, None, "parathas"), (None, None, "chai")]

    def test_quantity_only_chunk_attaches_to_previous_food(self):
        assert self.ex("leftover biryani, maybe two thirds of the box") == [(0.667, "box", "biryani")]

    def test_dish_names_with_separators_are_protected(self):
        assert self.ex("fish & chips and a beer") == [(None, None, "fish and chips"), (1, None, "beer")]
        assert self.ex("bagel with cream cheese") == [(None, None, "bagel with cream cheese")]

    def test_negations_are_dropped(self):
        assert [m[2] for m in self.ex("skipped lunch but had a samosa")] == ["samosa"]

    def test_meal_type(self):
        assert detect_meal_type("2 rotis for dinner") == "dinner"
        assert detect_meal_type("quick snack") == "snack"
        assert detect_meal_type("hello") is None


class TestProfile:
    def test_from_memories(self):
        p = profile_from_memories(
            [
                {"key": "dietary_restriction", "value": "vegetarian"},
                {"key": "allergy", "value": "allergic to peanuts and shellfish"},
                {"key": "note", "value": "lactose intolerant"},
            ]
        )
        assert p.diet == "vegetarian"
        assert {"peanut", "shellfish", "dairy"} <= p.allergens

    def test_negated_diet_is_ignored(self):
        assert profile_from_memories([{"key": "diet", "value": "not vegetarian anymore"}]).diet is None
        assert profile_from_memories([{"key": "diet", "value": "non-vegetarian"}]).diet is None
