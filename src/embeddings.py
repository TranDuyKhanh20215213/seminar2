from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        ...


class FakeEmbeddingModel(EmbeddingModel):
    """Deterministic, hash-seeded vectors. No downloads — used across the test suite."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.RandomState(seed)
            vectors.append(rng.rand(self.dim))
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim))


class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True)
