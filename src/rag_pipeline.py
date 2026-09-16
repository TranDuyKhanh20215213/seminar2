from typing import List, Optional

from src.generator import Generator
from src.models import Query, Recommendation
from src.retriever import VectorRetriever


class RAGRecommender:
    def __init__(self, retriever: VectorRetriever, generator: Generator, defenses: Optional[List] = None):
        self.retriever = retriever
        self.generator = generator
        self.defenses = defenses or []

    def recommend(self, query: Query, top_k: int = 5) -> Recommendation:
        retrieved = self.retriever.search(query.text, top_k=top_k)
        for defense in self.defenses:
            retrieved = defense.filter(query, retrieved)
        explanation = self.generator.generate(query.text, retrieved)
        ranked_items = [doc.item for doc in retrieved]
        return Recommendation(query=query, ranked_items=ranked_items, explanation=explanation)
