from typing import List

import numpy as np

from src.embeddings import EmbeddingModel
from src.models import RetrievedDocument
from src.repository.base import ItemRepository


class VectorRetriever:
    def __init__(self, repository: ItemRepository, embedding_model: EmbeddingModel):
        self.repository = repository
        self.embedding_model = embedding_model
        self._index_ids: List[str] = []
        self._index_vectors = np.zeros((0, 1))

    def build_index(self) -> None:
        items = self.repository.all()
        self._index_ids = [item.item_id for item in items]
        texts = [f"{item.title}. {item.description}" for item in items]
        self._index_vectors = self.embedding_model.encode(texts) if texts else np.zeros((0, 1))

    def search(self, query_text: str, top_k: int = 5) -> List[RetrievedDocument]:
        if len(self._index_ids) == 0:
            return []
        query_vector = self.embedding_model.encode([query_text])[0]
        similarities = self._cosine_similarity(self._index_vectors, query_vector)
        ranked_indices = np.argsort(-similarities)[:top_k]
        return [
            RetrievedDocument(item=self.repository.get(self._index_ids[i]), score=float(similarities[i]))
            for i in ranked_indices
        ]

    @staticmethod
    def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        vector_norm = vector / (np.linalg.norm(vector) + 1e-10)
        return matrix_norm @ vector_norm
