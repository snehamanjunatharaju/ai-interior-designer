"""
Embedding generation.

Uses a deterministic hashing-based embedder as the default so the system has
zero external dependencies and works fully offline. When
`sentence-transformers` is installed it is used automatically for markedly
better semantic quality, selectable via EMBEDDING_BACKEND.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

import numpy as np

from app.core.config import settings
from app.core.observability import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Expanded stopword list: interior-design text is short and domain-specific,
# so aggressive stopword removal measurably improves retrieval precision.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "this", "these", "those", "use", "using",
    "used", "can", "also", "more", "than", "then", "they", "their", "them",
    "we", "you", "your", "our", "not", "but", "if", "so", "such", "which",
    "when", "where", "while", "who", "what", "how", "all", "any", "each",
}

# Domain synonym expansion: designers and clients use different vocabulary for
# the same concept, so queries are normalised onto a shared term space.
_SYNONYMS = {
    "sofa": ["couch", "settee", "loveseat"],
    "tv": ["television", "telly", "media"],
    "bed": ["mattress", "bedframe"],
    "light": ["lighting", "lamp", "luminaire", "illumination"],
    "color": ["colour", "palette", "shade", "paint"],
    "storage": ["cupboard", "wardrobe", "closet", "shelf", "shelving"],
    "table": ["desk", "worktable", "console"],
    "chair": ["seating", "stool", "armchair", "seating"],
    "rug": ["carpet", "mat", "rugs"],
    "plant": ["plants", "greenery", "foliage", "indoor"],
    "kitchen": ["galley", "pantry", "countertop", "counter"],
    "bathroom": ["washroom", "restroom", "toilet", "bath"],
    "balcony": ["terrace", "verandah", "deck", "patio"],
    "vastu": ["vastu", "vaastu", "orientation", "compass"],
    "small": ["compact", "tiny", "narrow", "cramped"],
    "bright": ["light-filled", "airy", "sunny", "well-lit"],
    "cheap": ["affordable", "budget", "low-cost", "economical"],
    "luxury": ["premium", "high-end", "luxurious", "upscale"],
    "kids": ["child", "children", "toddler", "kids"],
    "work": ["office", "study", "workspace", "wfh"],
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


def _expand(tokens: Sequence[str]) -> List[str]:
    """Expand tokens with known synonyms, preserving original weighting."""
    out = list(tokens)
    for token in tokens:
        out.extend(_SYNONYMS.get(token, []))
    return out


class HashingEmbedder:
    """
    Deterministic feature-hashing embedder with L2 normalization.

    Maps tokens into a fixed-dimensional space using signed hashing, then
    applies sublinear TF weighting. This is the classic hashing-trick
    approach: no vocabulary file, no model download, and identical output
    across processes, which makes caching and reproducibility trivial.
    """

    def __init__(self, dim: int = None) -> None:
        self.dim = dim or settings.EMBEDDING_DIM
        self.name = f"hashing-{self.dim}"

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return idx, sign

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _expand(_tokenize(text))
        if not tokens:
            return vec

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            idx, sign = self._hash(token)
            # Sublinear TF damps repeated tokens.
            vec[idx] += sign * (1.0 + math.log(count))

        # Bigram signal captures local phrase order ("reading nook").
        for a, b in zip(tokens, tokens[1:]):
            idx, sign = self._hash(f"{a}_{b}")
            vec[idx] += sign * 0.5

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


class SentenceTransformerEmbedder:
    """Higher-quality embedder backed by sentence-transformers, if present."""

    def __init__(self, model_name: str = None) -> None:
        from sentence_transformers import SentenceTransformer  # local import

        self.model = SentenceTransformer(model_name or settings.EMBEDDING_MODEL)
        self.name = f"st-{model_name or settings.EMBEDDING_MODEL}"
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(text), dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self.model.encode(list(texts)), dtype=np.float32)


_embedder = None


def get_embedder():
    """Return the configured embedder, falling back to hashing if needed."""
    global _embedder
    if _embedder is not None:
        return _embedder

    backend = __import__("os").getenv("EMBEDDING_BACKEND", "auto").lower()
    if backend in {"auto", "sentence-transformers", "st"}:
        try:
            _embedder = SentenceTransformerEmbedder()
            logger.info("embedder_initialized backend=%s", _embedder.name)
            return _embedder
        except Exception as exc:
            if backend != "auto":
                raise
            logger.info("sentence-transformers unavailable (%s), using hashing", exc)

    _embedder = HashingEmbedder()
    logger.info("embedder_initialized backend=%s", _embedder.name)
    return _embedder
