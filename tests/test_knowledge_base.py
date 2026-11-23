"""
Data-quality tests for the nutrition knowledge base. The KB is code: it is versioned,
reviewed and tested like code.
"""

from backend.rag.corpus import ALLERGENS, DIETS, SOURCES, default_corpus
from backend.rag.retriever import HybridRetriever
from backend.rag.text import canonical_key

corpus = default_corpus()


def test_size_and_unique_ids():
    assert len(corpus) >= 400
    assert len({d.id for d in corpus.docs}) == len(corpus)


def test_enums_and_provenance():
    for d in corpus.docs:
        assert d.diet in DIETS, d.id
        assert set(d.allergens) <= set(ALLERGENS), d.id
        assert d.source in SOURCES, d.id
        assert d.serving_qty > 0 and d.serving_g > 0, d.id
        assert all(v >= 0 for v in d.per_serving().values()), d.id


def test_energy_is_consistent_with_macros():
    """Atwater check: kcal ~ 4*protein + 4*carbs + 9*fat (alcohol is exempt: 7 kcal/g ethanol)."""
    bad = []
    for d in corpus.docs:
        if d.category == "alcohol":
            continue
        estimate = 4 * d.protein_g + 4 * d.carbs_g + 9 * d.fat_g
        if abs(estimate - d.kcal) > max(0.2 * d.kcal, 15):
            bad.append((d.id, d.kcal, round(estimate)))
    assert not bad, f"energy/macro mismatch: {bad}"


def test_no_surface_form_points_at_two_foods():
    owner: dict[str, str] = {}
    clashes = []
    for d in corpus.docs:
        for s in d.surface_forms:
            key = canonical_key(s)
            if owner.setdefault(key, d.id) != d.id:
                clashes.append((s, owner[key], d.id))
    assert not clashes, clashes


def test_every_food_is_retrievable_by_its_own_name():
    """Self-retrieval over the whole KB: each food's name returns that food first."""
    retriever = HybridRetriever(corpus)
    misses = [d.id for d in corpus.docs if retriever.search(d.name, k=1).top.doc.id != d.id]
    assert not misses, misses
