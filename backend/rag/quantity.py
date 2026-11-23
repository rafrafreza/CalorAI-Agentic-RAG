"""
Quantity and unit parsing for free-text food mentions.

    "2 parathas"                 -> (2.0,  None,    "parathas")
    "half a plate of biryani"    -> (0.5,  "plate", "biryani")
    "two thirds of the box"      -> (0.667,"box",   "")
    "200g grilled chicken"       -> (200,  "g",     "grilled chicken")
    "1 1/2 cups dal"             -> (1.5,  "cup",   "dal")
    "a couple of eggs"           -> (2.0,  None,    "eggs")
    "some rice"                  -> (None, None,    "rice")   vague=True
"""

from __future__ import annotations

import re
from dataclasses import dataclass

UNICODE_FRACTIONS = {"½": 0.5, "⅓": 1 / 3, "⅔": 2 / 3, "¼": 0.25, "¾": 0.75, "⅛": 0.125}

NUMBER_WORDS: dict[str, float] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "dozen": 12,
    "couple": 2,
    "pair": 2,
    "half": 0.5,
    "quarter": 0.25,
    "double": 2,
}
FRACTION_WORDS: dict[str, float] = {
    "half": 2,
    "halves": 2,
    "third": 3,
    "thirds": 3,
    "quarter": 4,
    "quarters": 4,
    "fourth": 4,
    "fourths": 4,
}
VAGUE_WORDS = frozenset({"some", "few", "several", "bunch", "lots", "lot", "bit", "little", "many", "much"})
FILLERS = frozenset(
    {
        "about",
        "around",
        "approximately",
        "approx",
        "roughly",
        "like",
        "maybe",
        "almost",
        "nearly",
        "probably",
        "leftover",
        "leftovers",
        "another",
        "more",
        "extra",
        "my",
        "the",
        "of",
        "just",
        "only",
        "actually",
        "also",
        "basically",
        "full",
        "whole",
    }
)

UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    "g": ("g", "gm", "gms", "gram", "grams", "gr", "grm", "grms"),
    "kg": ("kg", "kgs", "kilo", "kilos", "kilogram", "kilograms"),
    "oz": ("oz", "ounce", "ounces"),
    "lb": ("lb", "lbs", "pound", "pounds"),
    "ml": ("ml", "mls", "milliliter", "milliliters", "millilitre", "millilitres"),
    "l": ("l", "litre", "litres", "liter", "liters", "ltr"),
    "cup": ("cup", "cups", "mug", "mugs"),
    "tbsp": ("tbsp", "tbs", "tablespoon", "tablespoons", "tbsps"),
    "tsp": ("tsp", "teaspoon", "teaspoons", "tsps"),
    "glass": ("glass", "glasses"),
    "bowl": ("bowl", "bowls"),
    "katori": ("katori", "katoris"),
    "plate": ("plate", "plates", "platter", "platters"),
    "piece": ("piece", "pieces", "pc", "pcs", "nos"),
    "slice": ("slice", "slices"),
    "serving": ("serving", "servings", "portion", "portions", "helping", "helpings"),
    "scoop": ("scoop", "scoops"),
    "box": ("box", "boxes", "container", "containers", "tub", "tubs"),
    "can": ("can", "cans", "tin", "tins"),
    "bottle": ("bottle", "bottles"),
    "handful": ("handful", "handfuls"),
    "packet": ("packet", "packets", "pack", "packs", "bag", "bags", "sachet", "sachets"),
    "bar": ("bar", "bars"),
    "shot": ("shot", "shots"),
    "pint": ("pint", "pints"),
}
UNIT_INDEX: dict[str, str] = {alias: unit for unit, aliases in UNIT_ALIASES.items() for alias in aliases}
# Single letters are only units right after a number ("200 g"), never on their own ("a g...").
_AMBIGUOUS_UNIT_TOKENS = frozenset({"g", "l"})

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?|\d+(?:\.\d+)?|[½⅓⅔¼¾⅛]|[a-z]+(?:'[a-z]+)?")
_NUM_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)([a-z]+)\b")
_TRAILING_QTY_RE = re.compile(r"(?:\bx\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*x)\s*$")


@dataclass(frozen=True)
class ParsedQuantity:
    quantity: float | None
    unit: str | None
    food: str
    vague: bool = False

    def as_text(self) -> str:
        if self.quantity is None:
            return ""
        q = f"{self.quantity:.2f}".rstrip("0").rstrip(".")
        return f"{q} {self.unit}" if self.unit else q


_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")
_PERCENT_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*%")


def _as_number(tok: str) -> float | None:
    """Digits, decimals, fractions and unicode fractions only ("nan" is a bread, not a number)."""
    if tok in UNICODE_FRACTIONS:
        return UNICODE_FRACTIONS[tok]
    if not _NUMERIC_RE.fullmatch(tok):
        return None
    if "/" in tok:
        num, den = tok.split("/", 1)
        try:
            return float(num) / float(den) if float(den) else None
        except ValueError:
            return None
    try:
        return float(tok)
    except ValueError:
        return None


def _tokenize(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    # "200g" -> "200 g", "2x" stays handled by the trailing-quantity regex.
    text = _NUM_UNIT_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}" if m.group(2) in UNIT_INDEX else m.group(0), text)
    text = re.sub(r"(\d)([½⅓⅔¼¾⅛])", r"\1 \2", text)
    return _TOKEN_RE.findall(text)


def parse_quantity(text: str) -> ParsedQuantity:
    """Parse a leading (or trailing ``x3``) quantity + unit from a food mention."""
    raw = (text or "").strip().lower()
    if _PERCENT_RE.match(raw):  # "2% milk" names a product, it is not a quantity
        return ParsedQuantity(quantity=None, unit=None, food=raw)
    trailing = _TRAILING_QTY_RE.search(raw)
    trailing_qty = None
    if trailing:
        trailing_qty = float(trailing.group(1) or trailing.group(2))
        raw = raw[: trailing.start()].strip()

    toks = _tokenize(raw)
    i, n = 0, len(toks)
    qty: float | None = None
    vague = False

    def peek(k: int = 0) -> str:
        return toks[i + k] if i + k < n else ""

    while i < n and toks[i] in FILLERS and toks[i] not in NUMBER_WORDS:
        i += 1

    if (i < n and toks[i] in VAGUE_WORDS) or (peek() in {"a", "an"} and peek(1) in VAGUE_WORDS):
        vague = True
        while i < n and (toks[i] in VAGUE_WORDS or toks[i] in {"a", "an", "of"}):
            i += 1
    elif i < n and (num := _as_number(toks[i])) is not None:
        qty = num
        i += 1
        if (frac := _as_number(peek())) is not None and frac < 1:  # "1 1/2", "1 ½"
            qty += frac
            i += 1
        elif peek() in FRACTION_WORDS and qty >= 1:  # "2 thirds"
            qty = qty / FRACTION_WORDS[peek()]
            i += 1
    elif i < n and toks[i] in NUMBER_WORDS:
        word = toks[i]
        i += 1
        if word == "half":
            qty = 0.5
            if peek() in {"a", "an"}:
                i += 1
        elif word in {"a", "an"} and peek() == "half":
            qty = 0.5
            i += 1
        elif word in {"a", "an"} and peek() in {"couple", "pair", "dozen"}:
            qty = NUMBER_WORDS[peek()]
            i += 1
        elif peek() in FRACTION_WORDS and word not in {"quarter", "half"}:  # "two thirds", "a quarter"
            qty = NUMBER_WORDS[word] / FRACTION_WORDS[peek()]
            i += 1
        elif peek() == "and" and peek(1) in {"a", "an"} and peek(2) == "half":  # "one and a half"
            qty = NUMBER_WORDS[word] + 0.5
            i += 3
        else:
            qty = NUMBER_WORDS[word]
        if peek() == "of" and qty is not None:
            i += 1

    unit: str | None = None
    while i < n and toks[i] in {"of", "the", "a", "an", "x"}:
        i += 1
    if i < n and toks[i] in UNIT_INDEX and (qty is not None or toks[i] not in _AMBIGUOUS_UNIT_TOKENS):
        # A unit word with no number ("bowl of dal") means one of that unit.
        unit = UNIT_INDEX[toks[i]]
        if qty is None and not vague:
            qty = 1.0
        i += 1
        while i < n and toks[i] in {"of", "the", "a", "an"}:
            i += 1

    rest = [t for t in toks[i:] if t not in {"of", "the"}]
    while rest and rest[0] in FILLERS:
        rest = rest[1:]
    food = " ".join(rest).strip()

    if trailing_qty is not None and qty is None:
        qty = trailing_qty
    return ParsedQuantity(quantity=round(qty, 3) if qty is not None else None, unit=unit, food=food, vague=vague)
