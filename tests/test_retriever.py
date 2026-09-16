from src.embeddings import FakeEmbeddingModel
from src.models import Item
from src.repository.memory_repository import InMemoryItemRepository
from src.retriever import VectorRetriever


def make_retriever():
    repository = InMemoryItemRepository()
    repository.add(Item(item_id="i1", title="Wireless Mouse", description="Ergonomic wireless mouse"))
    repository.add(Item(item_id="i2", title="Mechanical Keyboard", description="Quiet mechanical keyboard"))
    repository.add(Item(item_id="i3", title="Mechanical Keyboard", description="Quiet mechanical keyboard"))
    retriever = VectorRetriever(repository, FakeEmbeddingModel(dim=8))
    retriever.build_index()
    return retriever


def test_search_returns_at_most_top_k_results():
    retriever = make_retriever()
    results = retriever.search("keyboard", top_k=2)
    assert len(results) == 2


def test_search_results_are_sorted_by_descending_score():
    retriever = make_retriever()
    results = retriever.search("keyboard", top_k=3)
    scores = [doc.score for doc in results]
    assert scores == sorted(scores, reverse=True)


def test_identical_items_get_identical_scores_for_same_query():
    retriever = make_retriever()
    results = {doc.item.item_id: doc.score for doc in retriever.search("keyboard", top_k=3)}
    assert results["i2"] == results["i3"]


def test_search_on_empty_repository_returns_empty_list():
    repository = InMemoryItemRepository()
    retriever = VectorRetriever(repository, FakeEmbeddingModel(dim=8))
    retriever.build_index()
    assert retriever.search("anything") == []
