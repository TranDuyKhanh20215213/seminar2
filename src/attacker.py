from typing import List

import numpy as np

from src.embeddings import EmbeddingModel
from src.models import Item
from src.repository.base import ItemRepository


class PoisonRAGAttacker:
    """Crafts adversarial catalog items whose embedding is optimized to be
    retrieved for a target set of user queries, following the Poison-RAG
    data-poisoning strategy: greedily append vocabulary terms that increase
    cosine similarity to the target query centroid while keeping the
    attacker-controlled promotional payload intact."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def craft_adversarial_item(
        self,
        target_item_id: str,
        promotional_text: str,
        seed_queries: List[str],
        vocabulary: List[str],
        iterations: int = 20,
    ) -> Item:
        target_vector = self.embedding_model.encode(seed_queries).mean(axis=0)

        best_text = promotional_text
        best_score = self._similarity(best_text, target_vector)

        rng = np.random.RandomState(0)
        for _ in range(iterations):
            candidate_word = vocabulary[rng.randint(len(vocabulary))]
            candidate_text = f"{best_text} {candidate_word}"
            candidate_score = self._similarity(candidate_text, target_vector)
            if candidate_score > best_score:
                best_text, best_score = candidate_text, candidate_score

        return Item(
            item_id=f"poison-{target_item_id}",
            title=f"Sponsored: {target_item_id}",
            description=best_text,
            metadata={
                "is_adversarial": True,
                "target_item_id": target_item_id,
                "attack_score": best_score,
            },
        )

    def _similarity(self, text: str, target_vector: np.ndarray) -> float:
        vector = self.embedding_model.encode([text])[0]
        denom = (np.linalg.norm(vector) * np.linalg.norm(target_vector)) + 1e-10
        return float(vector @ target_vector / denom)

    def inject(self, repository: ItemRepository, adversarial_items: List[Item]) -> None:
        for item in adversarial_items:
            repository.add(item)
