"""
Deterministic portion arithmetic: (food, quantity, unit) -> scaled nutrition.

The LLM never does this math. It decides *what* was eaten; this module decides
*how much energy that is*, and records every assumption it had to make so the UI
can show it ("1 box ~ 1.5 servings", "assumed density ~1 g/ml").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.rag.corpus import MACROS, FoodDoc

MASS_G: dict[str, float] = {"g": 1.0, "kg": 1000.0, "oz": 28.35, "lb": 453.6}
VOLUME_ML: dict[str, float] = {
    "ml": 1.0,
    "l": 1000.0,
    "cup": 240.0,
    "tbsp": 15.0,
    "tsp": 5.0,
    "glass": 250.0,
    "can": 330.0,
    "bottle": 500.0,
    "shot": 44.0,
    "pint": 473.0,
}
# Count-like units that describe "one of the thing" (unit omitted -> count in these).
COUNT_UNITS = frozenset({"piece", "slice", "scoop", "bar", "packet", "shot"})
# Containers: how many reference servings one container usually holds.
CONTAINER_SERVINGS: dict[str, float] = {"bowl": 1.0, "katori": 0.75, "plate": 1.0, "box": 1.5, "serving": 1.0}
HANDFUL_G = 30.0


@dataclass
class Portion:
    factor: float  # multiplier applied to the per-serving values
    grams: float
    assumptions: list[str] = field(default_factory=list)


@dataclass
class ScaledNutrition:
    kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    grams: float
    factor: float
    assumptions: list[str]

    def as_dict(self) -> dict:
        return {
            "kcal": self.kcal,
            "protein_g": self.protein_g,
            "carbs_g": self.carbs_g,
            "fat_g": self.fat_g,
            "fiber_g": self.fiber_g,
            "grams": self.grams,
        }


def _fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def resolve_portion(doc: FoodDoc, quantity: float | None, unit: str | None) -> Portion:
    assumptions: list[str] = []
    counted = quantity is not None
    if quantity is None:
        quantity = 1.0 if doc.serving_unit not in COUNT_UNITS else doc.serving_qty
        assumptions.append(f"Quantity not given; assumed one serving ({doc.serving_label}).")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    unit = (unit or "").strip().lower() or None
    s_unit, s_qty, s_g = doc.serving_unit, doc.serving_qty, doc.serving_g

    if counted and doc.piece_g and s_unit in MASS_G and (unit is None or unit in COUNT_UNITS):
        # Per-gram entry counted in pieces: "10 almonds", "2 chicken breasts".
        grams = quantity * doc.piece_g
        assumptions.append(f"1 piece of {doc.name.lower()} ~ {doc.piece_g:g} g.")
        return Portion(factor=grams / s_g, grams=grams, assumptions=assumptions)

    if unit is None:
        if s_unit in COUNT_UNITS or s_unit in VOLUME_ML or s_unit in CONTAINER_SERVINGS:
            # "2 parathas" -> 2 pieces; "2 chai" -> 2 cups; "2 dal" -> 2 cups.
            factor = quantity / s_qty if s_unit in COUNT_UNITS else quantity
        else:  # per-100 g style entries: a bare number means servings
            factor = quantity
            if quantity != 1:
                assumptions.append(f"Read {_fmt(quantity)} as servings of {doc.serving_label}.")
        return Portion(factor=factor, grams=factor * s_g, assumptions=assumptions)

    if unit == s_unit:
        factor = quantity / s_qty
        return Portion(factor=factor, grams=factor * s_g, assumptions=assumptions)

    if unit in MASS_G:
        grams = quantity * MASS_G[unit]
        return Portion(factor=grams / s_g, grams=grams, assumptions=assumptions)

    if unit in VOLUME_ML:
        ml = quantity * VOLUME_ML[unit]
        if s_unit in VOLUME_ML:  # both volumes: exact ratio, using serving_g as the reference volume
            ref_ml = s_g if s_unit != "ml" else s_qty
            factor = ml / ref_ml
        else:
            factor = ml / s_g
            assumptions.append("Converted volume to weight assuming ~1 g/ml.")
        return Portion(factor=factor, grams=factor * s_g, assumptions=assumptions)

    if unit == "handful":
        grams = quantity * HANDFUL_G
        assumptions.append(f"1 handful ~ {HANDFUL_G:g} g.")
        return Portion(factor=grams / s_g, grams=grams, assumptions=assumptions)

    if unit in CONTAINER_SERVINGS:
        per = CONTAINER_SERVINGS[unit]
        factor = quantity * per
        if unit != "serving":
            assumptions.append(f"1 {unit} ~ {_fmt(per)} serving ({doc.serving_label}).")
        return Portion(factor=factor, grams=factor * s_g, assumptions=assumptions)

    if unit in COUNT_UNITS:
        # e.g. "2 pieces of paneer tikka" when the serving is "6 pieces"
        factor = quantity / s_qty if s_unit in COUNT_UNITS else quantity
        if s_unit not in COUNT_UNITS:
            assumptions.append(f"Treated 1 {unit} as one serving ({doc.serving_label}).")
        return Portion(factor=factor, grams=factor * s_g, assumptions=assumptions)

    assumptions.append(f"Unrecognised unit '{unit}'; treated as servings.")
    return Portion(factor=quantity, grams=quantity * s_g, assumptions=assumptions)


def scale(doc: FoodDoc, quantity: float | None = None, unit: str | None = None) -> ScaledNutrition:
    portion = resolve_portion(doc, quantity, unit)
    values = {m: round(getattr(doc, m) * portion.factor, 1) for m in MACROS}
    return ScaledNutrition(
        **values,
        grams=round(portion.grams, 1),
        factor=round(portion.factor, 4),
        assumptions=portion.assumptions,
    )
