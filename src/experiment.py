import json
from dataclasses import dataclass
from typing import List

from src.attacker import PoisonRAGAttacker
from src.defenses import Defense
from src.embeddings import EmbeddingModel
from src.generator import Generator
from src.metrics import attack_success_rate
from src.models import Item, Query
from src.rag_pipeline import RAGRecommender
from src.repository.base import ItemRepository
from src.repository.memory_repository import InMemoryItemRepository
from src.retriever import VectorRetriever


@dataclass
class ExperimentConfig:
    """One named combination of generator, attack toggle, and defenses to evaluate."""
    name: str
    generator: Generator
    use_attack: bool
    defenses: List[Defense]


class ExperimentRunner:
    def __init__(self, base_repository: ItemRepository, embedding_model: EmbeddingModel, attacker: PoisonRAGAttacker):
        self.base_repository = base_repository
        self.embedding_model = embedding_model
        self.attacker = attacker

    def run(self, config: ExperimentConfig, queries: List[Query], target_item_id: str, adversarial_item: Item) -> dict:
        """Runs one experiment configuration against a fresh in-memory copy of the base catalog — the on-disk seed data is never mutated by injecting the adversarial item, and repeated calls with different configs never see each other's injected items."""
        repository = InMemoryItemRepository()
        for item in self.base_repository.all():
            repository.add(item)
        if config.use_attack:
            self.attacker.inject(repository, [adversarial_item])

        retriever = VectorRetriever(repository, self.embedding_model)
        retriever.build_index()
        pipeline = RAGRecommender(retriever, config.generator, defenses=config.defenses)

        recommendations = [pipeline.recommend(query) for query in queries]
        sample_explanation = recommendations[0].explanation[:200] if recommendations else ""
        return {
            "config": config.name,
            "attack_success_rate": attack_success_rate(recommendations, target_item_id),
            "num_queries": len(queries),
            "sample_explanation": sample_explanation,
        }

    def save_results(self, results: List[dict], file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
