from typing import List, Optional

from src.generator import Generator
from src.models import Query, Recommendation
from src.retriever import VectorRetriever


class RAGRecommender:
    def __init__(
        self,
        retriever: VectorRetriever,
        generator: Generator,
        defenses: Optional[List] = None,
        overfetch_multiplier: int = 4,
    ):
        self.retriever = retriever
        self.generator = generator
        self.defenses = defenses or []
        self.overfetch_multiplier = overfetch_multiplier

    def recommend(self, query: Query, top_k: int = 5) -> Recommendation:
        fetch_k = top_k * self.overfetch_multiplier if self.defenses else top_k
        retrieved = self.retriever.search(query.text, top_k=fetch_k)
        for defense in self.defenses:
            retrieved = defense.filter(query, retrieved)
        retrieved = retrieved[:top_k]
        explanation = self.generator.generate(query.text, retrieved)
        ranked_items = [doc.item for doc in retrieved]
        return Recommendation(query=query, ranked_items=ranked_items, explanation=explanation)
