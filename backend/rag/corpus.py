"""
The nutrition knowledge base: a curated, versioned CSV of foods.

Every row carries its own provenance (``source``) so that numbers shown to the
user can always be traced back to where they came from:

* ``usda-ref``   - approximate USDA FoodData Central reference values (single foods)
* ``recipe-avg`` - typical home/restaurant recipe average (mixed dishes vary by cook)
* ``label-avg``  - typical packaged-product label average

Values are reference approximations for a consumer logging app, not medical-grade data.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "foods.csv"

SOURCES: dict[str, str] = {
    "usda-ref": "USDA FoodData Central reference values (approx.)",
    "recipe-avg": "Typical recipe average",
    "label-avg": "Typical packaged-product label average",
}
DIETS: tuple[str, ...] = ("vegan", "vegetarian", "eggetarian", "pescatarian", "non-veg")
ALLERGENS: tuple[str, ...] = (
    "dairy",
    "gluten",
    "egg",
    "peanut",
    "tree_nut",
    "soy",
    "fish",
    "shellfish",
    "sesame",
)
MACROS: tuple[str, ...] = ("kcal", "protein_g", "carbs_g", "fat_g", "fiber_g")


@dataclass(frozen=True)
class FoodDoc:
    """One knowledge-base entry. Nutrition values are *per serving*."""

    id: str
    name: str
    aliases: tuple[str, ...]
    description: str
    category: str
    cuisine: str
    serving_qty: float
    serving_unit: str
    serving_g: float
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    diet: str
    allergens: tuple[str, ...]
    source: str
    prior: float = 0.0
    piece_g: float | None = None  # weight of one countable piece for per-gram entries (1 almond ~ 1.2 g)
    index: int = field(default=-1, compare=False)

    @property
    def surface_forms(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @property
    def serving_label(self) -> str:
        qty = f"{self.serving_qty:g}"
        unit = self.serving_unit
        if unit in {"g", "ml"}:
            return f"{qty} {unit}"
        return f"{qty} {unit} (~{self.serving_g:g} g)"

    @property
    def source_label(self) -> str:
        return SOURCES.get(self.source, self.source)

    @cached_property
    def search_text(self) -> str:
        """Text indexed by the lexical channel (name/aliases weighted by repetition)."""
        parts = [self.name] * 3 + list(self.aliases) * 2 + [self.description, self.cuisine, self.category]
        return " ".join(parts)

    @cached_property
    def dense_text(self) -> str:
        """Natural-language text embedded by the dense channel."""
        alias = f" Also known as: {', '.join(self.aliases)}." if self.aliases else ""
        return f"{self.name}.{alias} {self.description}. Cuisine: {self.cuisine}. Category: {self.category}."

    def per_serving(self) -> dict[str, float]:
        return {m: getattr(self, m) for m in MACROS}

    def citation(self) -> dict[str, str]:
        return {"kb_id": self.id, "name": self.name, "source": self.source, "source_label": self.source_label}


def _split(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in (value or "").split("|") if v.strip())


def _row_to_doc(row: dict[str, str], index: int) -> FoodDoc:
    return FoodDoc(
        id=row["id"].strip(),
        name=row["name"].strip(),
        aliases=_split(row["aliases"]),
        description=row["description"].strip(),
        category=row["category"].strip(),
        cuisine=row["cuisine"].strip(),
        serving_qty=float(row["serving_qty"]),
        serving_unit=row["serving_unit"].strip(),
        serving_g=float(row["serving_g"]),
        kcal=float(row["kcal"]),
        protein_g=float(row["protein_g"]),
        carbs_g=float(row["carbs_g"]),
        fat_g=float(row["fat_g"]),
        fiber_g=float(row["fiber_g"]),
        diet=row["diet"].strip(),
        allergens=_split(row["allergens"]),
        source=row["source"].strip(),
        prior=float(row.get("prior") or 0),
        piece_g=float(row["piece_g"]) if (row.get("piece_g") or "").strip() else None,
        index=index,
    )


@dataclass(frozen=True)
class Corpus:
    docs: tuple[FoodDoc, ...]
    by_id: dict[str, FoodDoc]
    fingerprint: str

    def __len__(self) -> int:
        return len(self.docs)

    def get(self, food_id: str | None) -> FoodDoc | None:
        return self.by_id.get(food_id) if food_id else None

    def stats(self) -> dict:
        by_cuisine: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for d in self.docs:
            by_cuisine[d.cuisine] = by_cuisine.get(d.cuisine, 0) + 1
            by_source[d.source] = by_source.get(d.source, 0) + 1
        return {
            "foods": len(self.docs),
            "surface_forms": sum(len(d.surface_forms) for d in self.docs),
            "by_cuisine": dict(sorted(by_cuisine.items(), key=lambda kv: -kv[1])),
            "by_source": by_source,
            "fingerprint": self.fingerprint,
        }


def load_corpus(path: Path | str = DATA_PATH) -> Corpus:
    path = Path(path)
    raw = path.read_bytes()
    with path.open(newline="", encoding="utf-8") as fh:
        docs = tuple(_row_to_doc(row, i) for i, row in enumerate(csv.DictReader(fh)))
    by_id = {d.id: d for d in docs}
    if len(by_id) != len(docs):
        raise ValueError("Duplicate food ids in knowledge base")
    return Corpus(docs=docs, by_id=by_id, fingerprint=hashlib.sha1(raw).hexdigest()[:12])


@lru_cache(maxsize=1)
def default_corpus() -> Corpus:
    return load_corpus(DATA_PATH)
