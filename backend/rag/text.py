"""
Text primitives shared by every retrieval channel.

Food names are a hard retrieval domain: the same dish is romanised many ways
("paratha" / "parantha" / "prantha", "dal" / "daal" / "dhal", "chole" / "chhole"),
users type on phones (typos, plurals) and mix languages ("anda bhurji", "dahi").

Two views of a string are produced:

* :func:`terms`  - stop-word-filtered, lightly stemmed, phonetically folded tokens.
                   Used by BM25 and by the reranker's token-coverage features.
* :func:`fold`   - a phonetic folding of the whole string. Used to build the
                   character n-grams of the fuzzy channel.

Everything here is deterministic and dependency-free so it is cheap to unit test.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ASPIRATE = re.compile(r"([bcdgjkpstz])h+")
_DOUBLE = re.compile(r"([a-z])\1+")
_FINAL_I = re.compile(r"(?:ie|y)\b")


def _words(*lines: str) -> frozenset[str]:
    return frozenset(word for line in lines for word in line.split())


STOPWORDS: frozenset[str] = _words(
    "a an the of and or with without in on at to for from by into onto off over under",
    "some any my our your his her their this that these those it its there here",
    "i me we you he she they him them us is was were be been being am are",
    "had have has having ate eat eaten eating eats drank drinking drunk",
    "just also too very really bit little lot lots like about around approx approximately",
    "maybe kind sort quite pretty plus then than so but if as not no yes ok okay please",
    "today tonight yesterday now earlier later morning evening afternoon night",
    "leftover leftovers homemade freshly small big large medium regular mini tiny huge giant",
    "extra normal usual typical quick simple",
    "what when where which who whom whose why how do does did done can could should would will shall may might must",
)

# Words that must never lose a trailing "s".
_NO_STEM: frozenset[str] = frozenset(
    {"hummus", "couscous", "asparagus", "molasses", "swiss", "chaas", "khus", "octopus", "citrus"}
)


def normalize(text: str) -> str:
    """Lower-case, strip accents, map '&' to 'and', collapse non-alphanumerics to spaces."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ").replace("'", "")
    return _NON_ALNUM.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    """Normalised word tokens (no stop-word removal)."""
    norm = normalize(text)
    return norm.split() if norm else []


def stem(token: str) -> str:
    """
    Tiny plural stemmer, conservative on purpose (precision beats recall here).

    "-ies" plurals simply lose their "s" ("cookies" -> "cookie", "curries" -> "currie");
    :func:`fold` then maps a word-final "ie"/"y" to "i", so "curry" and "curries"
    (and "cookie"/"cookies") still meet in the same canonical term.
    """
    if len(token) <= 3 or token in _NO_STEM:
        return token
    if token.endswith("oes"):
        return token[:-2]  # tomatoes -> tomato
    if token.endswith(("ches", "shes", "xes", "sses", "zzes")):
        return token[:-2]  # sandwiches -> sandwich, glasses -> glass
    if token.endswith("s") and not token.endswith(("ss", "us")):
        return token[:-1]  # parathas -> paratha, eggs -> egg, idlis -> idli, rotis -> roti
    return token


def fold(text: str) -> str:
    """
    Phonetic folding for romanised South Asian (and general) food names.

    Applied identically to queries and documents, so it only needs to be
    consistent, not linguistically perfect:

        ee -> i, oo -> u            (keema/kima, aloo/alu)
        Ch, Dh, Bh, Kh, Th ... -> C, D, B, K, T  (aspirates: dhal/dal, chhole/chole)
        q -> k, z -> j, w -> v      (qeema/keema, zeera/jeera)
        doubled letters collapse    (daal/dal, chappati/chapati)
        word-final ie / y -> i      (idly/idli, curry/currie, maggie/maggi)
    """
    t = normalize(text)
    t = t.replace("ee", "i").replace("oo", "u")
    t = _ASPIRATE.sub(r"\1", t)
    t = t.replace("q", "k").replace("z", "j").replace("w", "v")
    t = _DOUBLE.sub(r"\1", t)
    return _FINAL_I.sub("i", t)


@lru_cache(maxsize=50_000)
def term(token: str) -> str:
    """Canonical form of a single token: stem first (plural on the raw spelling), then fold."""
    return fold(stem(token))


def terms(text: str, *, keep_numbers: bool = False) -> list[str]:
    """Content terms: tokens minus stop-words (and numbers), stemmed and folded."""
    out: list[str] = []
    for tok in tokens(text):
        if tok in STOPWORDS:
            continue
        if not keep_numbers and tok.isdigit():
            continue
        t = term(tok)
        if t:
            out.append(t)
    return out


def term_views(text: str) -> list[tuple[str, str]]:
    """(phonetic, orthographic) view of each content term: [("parata", "paratha"), ...]."""
    out: list[tuple[str, str]] = []
    for tok in tokens(text):
        if tok in STOPWORDS or tok.isdigit():
            continue
        raw = stem(tok)
        out.append((fold(raw), raw))
    return out


def canonical_key(text: str, *, keep_numbers: bool = True) -> str:
    """
    Order-preserving canonical key used for exact surface-form matching.

    Numbers are kept by default so that "chicken 65" and "chicken" stay distinct;
    callers strip them when a leading quantity is part of the query ("2 parathas").
    """
    return " ".join(terms(text, keep_numbers=keep_numbers))


def levenshtein(a: str, b: str, max_dist: int = 3) -> int:
    """Bounded Levenshtein distance (early exit once ``max_dist`` is exceeded)."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            row_min = min(row_min, cur[j])
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


@lru_cache(maxsize=200_000)
def token_similarity(a: str, b: str) -> float:
    """
    Similarity of two canonical terms in [0, 1].

    Exact matches score 1. Typos/transliterations score by normalised edit
    distance, but only for tokens long enough for that to be meaningful
    ("dal" vs "dam" must not match).
    """
    if a == b:
        return 1.0
    shortest = min(len(a), len(b))
    if shortest < 4:
        # Three-letter romanisations ("pao"/"pav", "dam"/"dum") get partial credit for
        # one substitution with the same initial; anything looser is noise.
        if shortest == 3 and len(a) == len(b) and a[0] == b[0] and levenshtein(a, b, 1) == 1:
            return 0.6
        return 0.0
    budget = 1 if max(len(a), len(b)) < 7 else 2
    dist = levenshtein(a, b, max_dist=budget)
    if dist > budget:
        return 0.0
    return 1.0 - dist / max(len(a), len(b))


def view_similarity(a: tuple[str, str], b: tuple[str, str]) -> float:
    """
    Best of phonetic and orthographic similarity. Folding is great for
    transliterations ("dhal" ~ "dal") but can hurt English typos ("oatmeel" folds
    towards "oatmilk"), so each view covers the other's blind spot.
    """
    return max(token_similarity(a[0], b[0]), token_similarity(a[1], b[1]))
