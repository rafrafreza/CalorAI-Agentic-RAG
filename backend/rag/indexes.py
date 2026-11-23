"""
Sparse retrieval indexes. Pure Python on purpose: the corpus is small (hundreds of
foods, ~2k surface forms), so an inverted index answers a query in well under a
millisecond without numpy/scipy, and the scoring is fully inspectable.

* :class:`BM25Index`      - Okapi BM25 over canonical terms (word level).
* :class:`CharNgramIndex` - TF-IDF over character trigrams of the phonetically
                            folded *surface forms* (name + each alias), max-pooled
                            per food. Robust to typos and transliteration variants.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from backend.rag.text import fold, terms


class BM25Index:
    """Okapi BM25 with the Lucene-style non-negative IDF."""

    def __init__(self, documents: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        doc_terms = [terms(text) for text in documents]
        self.n_docs = len(doc_terms)
        self.doc_len = [len(t) for t in doc_terms]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, toks in enumerate(doc_terms):
            for tok, tf in Counter(toks).items():
                self.postings[tok].append((doc_id, tf))
        self.idf = {
            tok: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for tok, p in self.postings.items()
        }

    def search(self, query: str, top_k: int = 25) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for tok in set(terms(query)):
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for doc_id, tf in self.postings[tok]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]


def char_ngrams(text: str, n: int = 3) -> Counter[str]:
    """Character n-grams of each folded word, padded with boundary markers."""
    grams: Counter[str] = Counter()
    for word in fold(text).split():
        padded = f" {word} "
        if len(padded) <= n:
            grams[padded] += 1
            continue
        for i in range(len(padded) - n + 1):
            grams[padded[i : i + n]] += 1
    return grams


class CharNgramIndex:
    """
    Cosine similarity over TF-IDF weighted character trigrams.

    Each food contributes several *surface forms* (its name and every alias); a
    food's score is the max over its forms. Indexing forms separately avoids
    diluting a strong alias match with the n-grams of unrelated aliases.
    """

    def __init__(self, surfaces: Iterable[tuple[int, str]], n: int = 3) -> None:
        self.n = n
        self.owner: list[int] = []
        raw: list[Counter[str]] = []
        for doc_id, text in surfaces:
            grams = char_ngrams(text, n)
            if grams:
                self.owner.append(doc_id)
                raw.append(grams)
        df: Counter[str] = Counter()
        for grams in raw:
            df.update(grams.keys())
        n_forms = len(raw)
        self.idf = {g: math.log((1 + n_forms) / (1 + c)) + 1 for g, c in df.items()}
        # Unseen query n-grams get the maximum IDF so they still count against the
        # query norm: "dragon fruit" must not look like a perfect match for "fruit".
        self.max_idf = math.log(1 + n_forms) + 1
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for form_id, grams in enumerate(raw):
            vec = self._weigh(grams)
            for g, w in vec.items():
                self.postings[g].append((form_id, w))

    def _weigh(self, grams: Counter[str]) -> dict[str, float]:
        vec = {g: (1 + math.log(tf)) * self.idf.get(g, self.max_idf) for g, tf in grams.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {g: w / norm for g, w in vec.items()}

    def search(self, query: str, top_k: int = 25) -> list[tuple[int, float]]:
        qvec = self._weigh(char_ngrams(query, self.n))
        form_scores: dict[int, float] = defaultdict(float)
        for g, qw in qvec.items():
            for form_id, w in self.postings.get(g, ()):
                form_scores[form_id] += qw * w
        best: dict[int, float] = {}
        for form_id, score in form_scores.items():
            doc_id = self.owner[form_id]
            if score > best.get(doc_id, 0.0):
                best[doc_id] = score
        return sorted(best.items(), key=lambda kv: -kv[1])[:top_k]
