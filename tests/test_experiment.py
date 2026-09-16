from src.attacker import PoisonRAGAttacker
from src.defenses import MetadataTrustDefense
from src.embeddings import FakeEmbeddingModel
from src.experiment import ExperimentConfig, ExperimentRunner
from src.generator import FakeGenerator
from src.models import Item, Query
from src.repository.memory_repository import InMemoryItemRepository


def make_runner():
    embedding_model = FakeEmbeddingModel(dim=8)
    base_repository = InMemoryItemRepository()
    base_repository.add(Item(item_id="i1", title="Mechanical Keyboard", description="Quiet mechanical keyboard"))
    attacker = PoisonRAGAttacker(embedding_model)
    return ExperimentRunner(base_repository, embedding_model, attacker)


def make_adversarial_item(runner):
    return runner.attacker.craft_adversarial_item(
        target_item_id="i1",
        promotional_text="Sponsored discount keyboard offer",
        seed_queries=["quiet mechanical keyboard"],
        vocabulary=["keyboard", "quiet", "office"],
        iterations=5,
    )


def test_run_without_attack_never_recommends_adversarial_item():
    runner = make_runner()
    adversarial_item = make_adversarial_item(runner)
    config = ExperimentConfig(name="baseline", generator=FakeGenerator(), use_attack=False, defenses=[])
    queries = [Query(query_id="q1", text="quiet mechanical keyboard")]

    result = runner.run(config, queries, target_item_id=adversarial_item.item_id, adversarial_item=adversarial_item)

    assert result["attack_success_rate"] == 0.0
    assert result["config"] == "baseline"


def test_run_with_attack_and_no_defense_can_succeed():
    runner = make_runner()
    adversarial_item = make_adversarial_item(runner)
    config = ExperimentConfig(name="attacked", generator=FakeGenerator(), use_attack=True, defenses=[])
    queries = [Query(query_id="q1", text="quiet mechanical keyboard")]

    result = runner.run(config, queries, target_item_id=adversarial_item.item_id, adversarial_item=adversarial_item)

    assert result["attack_success_rate"] == 1.0


def test_run_with_attack_and_metadata_trust_defense_blocks_attack():
    runner = make_runner()
    adversarial_item = make_adversarial_item(runner)
    config = ExperimentConfig(name="defended", generator=FakeGenerator(), use_attack=True, defenses=[MetadataTrustDefense()])
    queries = [Query(query_id="q1", text="quiet mechanical keyboard")]

    result = runner.run(config, queries, target_item_id=adversarial_item.item_id, adversarial_item=adversarial_item)

    assert result["attack_success_rate"] == 0.0


import json


def test_save_results_writes_valid_json(tmp_path):
    runner = make_runner()
    file_path = str(tmp_path / "results.json")
    runner.save_results([{"config": "baseline", "attack_success_rate": 0.0}], file_path)

    with open(file_path) as f:
        loaded = json.load(f)
    assert loaded[0]["config"] == "baseline"
