from src.attacker import PoisonRAGAttacker
from src.embeddings import FakeEmbeddingModel
from src.models import Item
from src.repository.memory_repository import InMemoryItemRepository


def test_craft_adversarial_item_is_flagged_and_targets_given_item():
    attacker = PoisonRAGAttacker(FakeEmbeddingModel(dim=8))
    item = attacker.craft_adversarial_item(
        target_item_id="i1",
        promotional_text="Buy the best deal now",
        seed_queries=["quiet mechanical keyboard", "ergonomic keyboard for office"],
        vocabulary=["keyboard", "office", "quiet", "ergonomic", "deal"],
        iterations=5,
    )
    assert item.metadata["is_adversarial"] is True
    assert item.metadata["target_item_id"] == "i1"
    assert "Buy the best deal now" in item.description


def test_craft_adversarial_item_increases_similarity_to_seed_queries():
    attacker = PoisonRAGAttacker(FakeEmbeddingModel(dim=8))
    seed_queries = ["quiet mechanical keyboard", "ergonomic keyboard for office"]
    baseline_score = attacker._similarity(
        "Buy the best deal now", FakeEmbeddingModel(dim=8).encode(seed_queries).mean(axis=0)
    )
    item = attacker.craft_adversarial_item(
        target_item_id="i1",
        promotional_text="Buy the best deal now",
        seed_queries=seed_queries,
        vocabulary=["keyboard", "office", "quiet", "ergonomic", "deal"],
        iterations=20,
    )
    assert item.metadata["attack_score"] >= baseline_score


def test_inject_adds_adversarial_items_to_repository():
    attacker = PoisonRAGAttacker(FakeEmbeddingModel(dim=8))
    repository = InMemoryItemRepository()
    adversarial_item = Item(item_id="poison-i1", title="Sponsored", description="desc", metadata={"is_adversarial": True})

    attacker.inject(repository, [adversarial_item])

    assert repository.get("poison-i1") is not None
    assert repository.get("poison-i1").metadata["is_adversarial"] is True
