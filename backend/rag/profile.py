"""
User dietary profile, derived deterministically from long-term memory.

The profile personalises retrieval in two ways:

1. **Disambiguation** - "biryani" for a vegetarian resolves to *Veg Biryani*, not the
   (more common) chicken variant, *unless* the user explicitly said "chicken".
2. **Safety** - allergen and diet conflicts are attached to every grounding result so
   the agent can warn ("satay usually contains peanuts") instead of silently logging.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from backend.rag.corpus import FoodDoc
from backend.rag.text import terms

# Diet -> diets of foods that conflict with it.
DIET_CONFLICTS: dict[str, frozenset[str]] = {
    "vegan": frozenset({"vegetarian", "eggetarian", "pescatarian", "non-veg"}),
    "vegetarian": frozenset({"pescatarian", "non-veg"}),
    "eggetarian": frozenset({"pescatarian", "non-veg"}),
    "pescatarian": frozenset({"non-veg"}),
}

# When disambiguating, which food diets are the most natural fit for a profile.
PREFERRED_DIETS: dict[str, frozenset[str]] = {
    "vegan": frozenset({"vegan"}),
    "vegetarian": frozenset({"vegetarian", "vegan"}),
    "eggetarian": frozenset({"vegetarian", "vegan", "eggetarian"}),
    "pescatarian": frozenset({"vegetarian", "vegan", "eggetarian", "pescatarian"}),
}

# Words that make an otherwise-conflicting choice *explicit* ("chicken biryani").
EXPLICIT_MARKERS: dict[str, frozenset[str]] = {
    "non-veg": frozenset(
        terms(
            "chicken mutton lamb beef pork goat gosht murgh keema kheema qeema meat bacon ham sausage "
            "pepperoni turkey steak salami nihari haleem kebab kabab shawarma doner gyro"
        )
    ),
    "pescatarian": frozenset(terms("fish prawn prawns shrimp salmon tuna machli meen crab sushi seafood")),
    "eggetarian": frozenset(terms("egg eggs anda omelette omelet")),
    "vegetarian": frozenset(terms("paneer cheese milk dairy ghee butter cream yogurt curd")),
}

_DIET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("vegan", re.compile(r"\bvegan\b")),
    ("pescatarian", re.compile(r"\bpesc[ae]tarian\b")),
    ("eggetarian", re.compile(r"\beggetarian\b")),
    ("vegetarian", re.compile(r"(?<!non[\s-])\bvegetarian\b|\bveggie\b|\bveg only\b|\bpure veg\b")),
]

_ALLERGEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "peanut": re.compile(r"\bpeanuts?\b|\bgroundnuts?\b|\bnut allerg"),
    "tree_nut": re.compile(r"\btree[\s-]?nuts?\b|\balmonds?\b|\bcashews?\b|\bwalnuts?\b|\bpistachios?\b|\bnut allerg"),
    "dairy": re.compile(r"\blactose\b|\bdairy\b|\bmilk allerg|\bcasein\b"),
    "gluten": re.compile(r"\bgluten\b|\bceliac\b|\bcoeliac\b|\bwheat allerg"),
    "egg": re.compile(r"\begg allerg|\ballergic to eggs?\b|\bno eggs?\b"),
    "soy": re.compile(r"\bsoy\b|\bsoya\b"),
    "fish": re.compile(r"\bfish allerg|\ballergic to fish\b"),
    "shellfish": re.compile(r"\bshellfish\b|\bprawns?\b|\bshrimps?\b|\bcrustacean"),
    "sesame": re.compile(r"\bsesame\b|\btahini\b"),
}
_ALLERGY_CONTEXT = re.compile(r"allerg|intoleran|can'?t (eat|have)|cannot (eat|have)|avoid|sensitiv|celiac|coeliac")


@dataclass(frozen=True)
class DietProfile:
    diet: str | None = None
    allergens: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_empty(self) -> bool:
        return self.diet is None and not self.allergens

    def conflicts(self, doc: FoodDoc) -> list[str]:
        out: list[str] = []
        if self.diet and doc.diet in DIET_CONFLICTS.get(self.diet, frozenset()):
            out.append(f"diet: {doc.name} is {doc.diet}; you're {self.diet}")
        for allergen in sorted(self.allergens.intersection(doc.allergens)):
            out.append(f"allergen: {doc.name} typically contains {allergen.replace('_', ' ')}")
        return out

    def is_explicit_choice(self, query_terms: Iterable[str], doc: FoodDoc) -> bool:
        """True if the query names the very attribute that conflicts (the user asked for it)."""
        q = set(query_terms)
        markers = EXPLICIT_MARKERS.get(doc.diet, frozenset())
        return bool(q & markers)

    def as_dict(self) -> dict:
        return {"diet": self.diet, "allergens": sorted(self.allergens)}


def profile_from_memories(memories: Iterable[dict]) -> DietProfile:
    """Build a profile from stored memory facts (any category; preferences matter most)."""
    diet: str | None = None
    allergens: set[str] = set()
    for m in memories:
        text = f"{m.get('key', '')} {m.get('value', '')}".lower().replace("_", " ")
        if diet is None:
            for name, pattern in _DIET_PATTERNS:
                if pattern.search(text) and not re.search(r"\bnot (a )?" + name, text):
                    diet = name
                    break
        if _ALLERGY_CONTEXT.search(text):
            for allergen, pattern in _ALLERGEN_PATTERNS.items():
                if pattern.search(text):
                    allergens.add(allergen)
    return DietProfile(diet=diet, allergens=frozenset(allergens))
