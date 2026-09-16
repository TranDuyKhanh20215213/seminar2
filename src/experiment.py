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
        repository = InMemoryItemRepository()
        for item in self.base_repository.all():
            repository.add(item)
        if config.use_attack:
            self.attacker.inject(repository, [adversarial_item])

        retriever = VectorRetriever(repository, self.embedding_model)
        retriever.build_index()
        pipeline = RAGRecommender(retriever, config.generator, defenses=config.defenses)

        recommendations = [pipeline.recommend(query) for query in queries]
        return {
            "config": config.name,
            "attack_success_rate": attack_success_rate(recommendations, target_item_id),
            "num_queries": len(queries),
        }
