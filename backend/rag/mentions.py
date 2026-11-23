"""
Food-mention extraction from a chat message - the front half of *speculative retrieval*.

    "had 2 parathas and chai for breakfast"
        -> meal_type="breakfast", mentions=[(2, None, "parathas"), (None, None, "chai")]

    "leftover biryani, maybe two thirds of the box"
        -> [(0.667, "box", "biryani")]        (a quantity-only chunk attaches to the previous food)

Multi-word dish names that contain a separator ("mac and cheese", "bagel with cream
cheese", "fish & chips") are protected using the knowledge base's own surface forms,
so they are not split into two foods.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from backend.rag.quantity import parse_quantity
from backend.rag.text import normalize, terms

_SEPARATORS = r"and|with|plus|along with|then|also|but|followed by|n"
_SPLIT_RE = re.compile(rf"\s*(?:[,;+]|\b(?:{_SEPARATORS})\b)\s*")
_SEP_WORD_RE = re.compile(rf"\b(?:{_SEPARATORS})\b")

_LEAD_IN_RE = re.compile(
    r"^(?:(?:i|we)\s+|i'?m\s+|i am\s+|we'?re\s+|we are\s+)?"
    r"(?:(?:just|also|then|finally|already|only)\s+)?"
    r"(?:had|have had|'?ve had|ate|eaten|eat|eating|having|have|drank|drink|drinking|grabbed|got|"
    r"consumed|finished|took|ordered|tried|snacked on|munched on|devoured|enjoyed|made)\s+"
)
_MEAL_TYPE_RE = re.compile(r"\b(breakfast|brunch|lunch|dinner|supper|snacks?|tiffin|dessert)\b")
_MEAL_CONTEXT_RE = re.compile(
    r"\b(?:for|at|during|in|as|with|after|before)\s+(?:my\s+|the\s+|a\s+|our\s+)?"
    r"(?:breakfast|brunch|lunch|dinner|supper|snacks?|tiffin|dessert|workout|gym)\b"
    r"|\b(?:this|in the|last|yesterday|today)\s+(?:morning|afternoon|evening|night)\b"
    r"|\b(?:today|tonight|yesterday|just now|earlier|right now|again|btw|lol|haha)\b"
)
_NEGATION_RE = re.compile(r"^(?:no|not|didn'?t|did not|skipped|skip|without|never|haven'?t|don'?t)\b")
_TRAILING_RE = re.compile(r"\s+(?:too|as well|also|please|thanks|thank you|only)$")
_PUNCT_RE = re.compile(r"[!?.:\"“”()\[\]{}]")

MEAL_TYPES = {
    "breakfast": "breakfast",
    "brunch": "breakfast",
    "lunch": "lunch",
    "dinner": "dinner",
    "supper": "dinner",
    "snack": "snack",
    "snacks": "snack",
    "tiffin": "snack",
    "dessert": "snack",
}


@dataclass(frozen=True)
class Mention:
    text: str
    food: str
    quantity: float | None
    unit: str | None
    vague: bool = False


def detect_meal_type(message: str) -> str | None:
    m = _MEAL_TYPE_RE.search(message.lower())
    return MEAL_TYPES.get(m.group(1)) if m else None


def protected_phrases(surface_forms: Iterable[str]) -> list[str]:
    """Normalised KB surface forms that contain a separator word (longest first)."""
    phrases = {normalize(s) for s in surface_forms}
    return sorted((p for p in phrases if " " in p and _SEP_WORD_RE.search(p)), key=len, reverse=True)


def extract_mentions(message: str, protected: list[str] | None = None) -> list[Mention]:
    text = _PUNCT_RE.sub(" ", (message or "").lower()).replace("&", " and ")
    text = _MEAL_CONTEXT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for phrase in protected or ():
        if phrase in text:
            text = re.sub(rf"\b{re.escape(phrase)}\b", phrase.replace(" ", "_"), text)

    mentions: list[Mention] = []
    for chunk in _SPLIT_RE.split(text):
        chunk = chunk.replace("_", " ").strip(" -")
        if not chunk or _NEGATION_RE.match(chunk):
            continue
        prev = None
        while prev != chunk:  # "i just had", "then had" ...
            prev, chunk = chunk, _LEAD_IN_RE.sub("", chunk).strip()
        chunk = _TRAILING_RE.sub("", chunk).strip()
        parsed = parse_quantity(chunk)
        food = parsed.food.strip()
        if not food or not terms(food):
            # "...biryani, maybe two thirds of the box": quantity-only chunk -> previous food
            if parsed.quantity is not None and mentions and mentions[-1].quantity is None:
                last = mentions[-1]
                mentions[-1] = Mention(last.text + ", " + chunk, last.food, parsed.quantity, parsed.unit, last.vague)
            continue
        mentions.append(Mention(chunk, food, parsed.quantity, parsed.unit, parsed.vague))
    return mentions
