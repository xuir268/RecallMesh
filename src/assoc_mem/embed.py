"""Embedding provider. Swappable; the engine never sees text."""
from __future__ import annotations

import numpy as np
import hashlib


class Embedder:
    """Protocol: encode(list[str]) -> (n, dim) float32, L2-normalised."""

    @property
    def dim(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic bag-of-tokens hashing. No model, no network, no API key.

    Good enough to exercise the pipeline and write tests against; replace with
    a real sentence encoder before any eval result means anything.
    """

    def __init__(self, dim: int = 128, seed: int = 0):
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int.from_bytes(hashlib.blake2b(f"{self._seed}:{tok}".encode(), digest_size=8).digest(), "little") % self._dim
                out[i, h] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return out
