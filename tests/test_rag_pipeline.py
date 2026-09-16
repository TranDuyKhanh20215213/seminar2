from src.embeddings import FakeEmbeddingModel
from src.generator import FakeGenerator
from src.models import Item, Query
from src.rag_pipeline import RAGRecommender
from src.repository.memory_repository import InMemoryItemRepository
from src.retriever import VectorRetriever


def make_pipeline(defenses=None):
    repository = InMemoryItemRepository()
    repository.add(Item(item_id="i1", title="Wireless Mouse", description="Ergonomic wireless mouse"))
    repository.add(Item(item_id="i2", title="Mechanical Keyboard", description="Quiet mechanical keyboard"))
    retriever = VectorRetriever(repository, FakeEmbeddingModel(dim=8))
    retriever.build_index()
    return RAGRecommender(retriever, FakeGenerator(), defenses=defenses)


def test_recommend_returns_ranked_items_from_retrieval():
    pipeline = make_pipeline()
    recommendation = pipeline.recommend(Query(query_id="q1", text="keyboard"), top_k=2)
    assert len(recommendation.ranked_items) == 2
    assert recommendation.query.query_id == "q1"


def test_recommend_explanation_comes_from_generator():
    pipeline = make_pipeline()
    recommendation = pipeline.recommend(Query(query_id="q1", text="keyboard"), top_k=1)
    assert "Recommended based on:" in recommendation.explanation


def test_recommend_applies_defenses_before_generation():
    class DropAllDefense:
        def filter(self, query, retrieved):
            return []

    pipeline = make_pipeline(defenses=[DropAllDefense()])
    recommendation = pipeline.recommend(Query(query_id="q1", text="keyboard"), top_k=2)
    assert recommendation.ranked_items == []
    assert recommendation.explanation == "Recommended based on: "
