"""Text normalisation, phonetic folding, quantity parsing and portion arithmetic."""

import pytest

from backend.rag.corpus import default_corpus
from backend.rag.portion import scale
from backend.rag.quantity import parse_quantity
from backend.rag.text import canonical_key, fold, levenshtein, stem, term_views, terms, token_similarity


class TestText:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("paratha", "parathas"),
            ("dal", "daal"),
            ("dal", "dhal"),
            ("chole", "chhole"),
            ("keema", "qeema"),
            ("idli", "idly"),
            ("jeera", "zeera"),
            ("aloo", "alu"),
            ("curry", "curries"),
            ("cookie", "cookies"),
            ("maggi", "maggie"),
            ("roti", "rotis"),
            ("tomato", "tomatoes"),
        ],
    )
    def test_variants_share_a_canonical_term(self, a, b):
        assert terms(a) == terms(b)

    def test_stopwords_and_numbers_are_dropped(self):
        assert terms("I had 2 of the leftover parathas today") == terms("parathas")

    def test_canonical_key_keeps_numbers_by_default(self):
        assert canonical_key("chicken 65") != canonical_key("chicken")
        assert canonical_key("2 parathas", keep_numbers=False) == canonical_key("paratha")

    def test_stem_is_conservative(self):
        assert stem("hummus") == "hummus"
        assert stem("glasses") == "glass"
        assert stem("sandwiches") == "sandwich"

    def test_fold_examples(self):
        assert fold("Chhole Bhature") == fold("chole bhature")
        assert fold("daal makhni") == fold("dal makhni")

    def test_levenshtein_bounded(self):
        assert levenshtein("kitten", "sitting") == 3
        assert levenshtein("abc", "abcdefgh", max_dist=2) == 3  # early exit returns max+1

    def test_token_similarity_guards_short_tokens(self):
        assert token_similarity("dal", "dal") == 1.0
        assert token_similarity("pao", "pav") == pytest.approx(0.6)
        assert token_similarity("tea", "pea") == 0.0
        assert token_similarity("biryani", "biriyani") > 0.8

    def test_term_views_pair_phonetic_and_orthographic(self):
        assert term_views("Oatmeel") == [("oatmil", "oatmeel")]


class TestQuantity:
    @pytest.mark.parametrize(
        "text,qty,unit,food",
        [
            ("2 parathas", 2, None, "parathas"),
            ("half a plate of biryani", 0.5, "plate", "biryani"),
            ("two thirds of the box", 0.667, "box", ""),
            ("200g grilled chicken", 200, "g", "grilled chicken"),
            ("1 1/2 cups dal", 1.5, "cup", "dal"),
            ("a couple of eggs", 2, None, "eggs"),
            ("one and a half cups rice", 1.5, "cup", "rice"),
            ("rotis x3", 3, None, "rotis"),
            ("a bowl of dal", 1, "bowl", "dal"),
            ("three quarters of a cup of milk", 0.75, "cup", "milk"),
            ("a dozen eggs", 12, None, "eggs"),
            ("chai", None, None, "chai"),
        ],
    )
    def test_parse(self, text, qty, unit, food):
        p = parse_quantity(text)
        assert p.quantity == (pytest.approx(qty) if qty is not None else None)
        assert p.unit == unit
        assert p.food == food

    def test_vague_quantities_are_flagged_not_guessed(self):
        p = parse_quantity("some rice")
        assert p.vague and p.quantity is None and p.food == "rice"

    def test_food_names_are_not_numbers(self):
        assert parse_quantity("nan").quantity is None  # "nan" is a bread (naan), not NaN
        assert parse_quantity("2% milk").food == "2% milk"


class TestPortion:
    corpus = default_corpus()

    def kcal(self, food_id, q=None, u=None):
        return scale(self.corpus.get(food_id), q, u).kcal

    def test_counts_in_natural_units(self):
        assert self.kcal("paratha", 2) == 520
        assert self.kcal("pancakes", 3) == pytest.approx(525)  # serving = 2 pancakes

    def test_mass_and_volume(self):
        assert self.kcal("chicken_breast", 200, "g") == pytest.approx(330)
        assert self.kcal("masala_chai", 200, "ml") == pytest.approx(120)
        assert self.kcal("whole_milk", 1, "glass") == pytest.approx(152.7, abs=0.1)

    def test_per_gram_foods_counted_in_pieces(self):
        # 10 almonds is ~12 g, not 10 x 28 g servings
        assert self.kcal("almonds", 10) == pytest.approx(70.3, abs=0.1)
        assert self.kcal("almonds") == pytest.approx(164)  # no quantity -> one serving

    def test_container_assumptions_are_recorded(self):
        n = scale(self.corpus.get("chicken_biryani"), 2 / 3, "box")
        assert n.kcal == pytest.approx(350, abs=1)
        assert any("box" in a for a in n.assumptions)

    def test_invalid_quantity(self):
        with pytest.raises(ValueError):
            scale(self.corpus.get("roti"), 0)
