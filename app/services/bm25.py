"""
BM25 lexical index for hybrid retrieval.

Why BM25 alongside embeddings
-----------------------------
Dense vectors are good at paraphrase and bad at exact terms. A brief asking
for a "bunk bed" needs the chunk that literally says "bunk bed"; a hashing
embedder (the default, because it needs no download) has no semantic
component at all, so lexical matching is doing most of the work regardless of
what the README claims. A branded product name or a number is even worse
served by dense retrieval, because those tokens are rare and the embedding
barely registers them.

BM25 is the standard fix: it rewards a term for being frequent in the
document but rare across the corpus, and it saturates term frequency so a
word repeated twenty times does not dominate. It is ~60 lines of arithmetic,
has no dependencies, and is fully deterministic, which matters for a system
whose selling point is reproducible output.

It is built here rather than pulled from a library so it can be tested
directly and so the fusion weights are visible.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

# Very common English words carry no signal and would flatten every score.
# Plain strings, not bytes: tokenize() produces str, and a bytes set would
# silently never match, leaving every stopword in the index.
STOPWORDS = frozenset("""
a an and are as at be been but by can could do does for from had has have
he her his if in into is it its me my of on or our out she should so than
that the their them then there these they this to up us was we were what
when where which who will with would you your
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# BM25 parameters. k1 controls term-frequency saturation, b controls how
# strongly long documents are penalised. These are the standard defaults and
# are rarely worth tuning on a corpus this small.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens, stopwords removed. Numbers are kept: a brief
    mentioning "90 cm" should match a passage that mentions the same."""
    return [
        t for t in _TOKEN_RE.findall((text or "").lower())
        if t not in STOPWORDS and len(t) > 1
    ]


class BM25Index:
    """An in-memory BM25 index over a fixed set of documents."""

    def __init__(self, k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self._docs: List[List[str]] = []
        self._doc_len: List[int] = []
        self._df: Counter = Counter()
        self._avg_len: float = 0.0
        self._idf: Dict[str, float] = {}

    def build(self, documents: Iterable[str]) -> "BM25Index":
        """Index the corpus. Safe to call repeatedly; it replaces the index."""
        self._docs = []
        self._doc_len = []
        self._df = Counter()

        for doc in documents:
            tokens = tokenize(doc)
            self._docs.append(tokens)
            self._doc_len.append(len(tokens))
            for term in set(tokens):
                self._df[term] += 1

        n = len(self._docs)
        self._avg_len = (sum(self._doc_len) / n) if n else 0.0

        # Standard BM25 inverse document frequency, with the +0.5 smoothing
        # that keeps the weight positive for a term present in every document.
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in self._df.items()
        }
        return self

    def score(self, query: str) -> List[Tuple[int, float]]:
        """Score every document against the query, returning (index, score)."""
        if not self._docs:
            return []
        terms = tokenize(query)
        if not terms:
            return []

        out: List[Tuple[int, float]] = []
        for i, doc in enumerate(self._docs):
            if not doc:
                continue
            counts = Counter(doc)
            length = self._doc_len[i] or 1
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * length / self._avg_len)
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                out.append((i, score))

        out.sort(key=lambda x: -x[1])
        return out

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        return self.score(query)[:k]

    @property
    def size(self) -> int:
        return len(self._docs)

    @property
    def vocabulary_size(self) -> int:
        return len(self._idf)


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Fuse several ranked lists into one using Reciprocal Rank Fusion.

    RRF is used rather than a weighted sum of normalised scores because the
    two signals are on incomparable scales: a hashing cosine sits around
    0.2 while BM25 is unbounded. RRF only cares about *position*, so a
    document ranked highly by either list wins, and no scale calibration is
    needed. It is the standard approach for hybrid retrieval for exactly this
    reason.

    `k` damps the influence of the very top ranks so a single list cannot
    dominate the result.
    """
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])
