from src.metrics import attack_success_rate, hit_rate_at_k, ndcg_at_k
from src.models import Item, Query, Recommendation


def make_recommendation(query_id, item_ids):
    return Recommendation(
        query=Query(query_id=query_id, text="q"),
        ranked_items=[Item(item_id=i, title=i, description="d") for i in item_ids],
        explanation="",
    )


def test_attack_success_rate_counts_recommendations_containing_target():
    recommendations = [
        make_recommendation("q1", ["poison-i1", "i2"]),
        make_recommendation("q2", ["i3", "i4"]),
    ]
    assert attack_success_rate(recommendations, target_item_id="poison-i1") == 0.5


def test_attack_success_rate_on_empty_list_is_zero():
    assert attack_success_rate([], target_item_id="poison-i1") == 0.0


def test_hit_rate_at_k_counts_queries_with_any_relevant_item_in_top_k():
    recommendations = [make_recommendation("q1", ["i1", "i2", "i3"])]
    relevant = {"q1": {"i3"}}
    assert hit_rate_at_k(recommendations, relevant, k=2) == 0.0
    assert hit_rate_at_k(recommendations, relevant, k=3) == 1.0


def test_ndcg_at_k_rewards_relevant_items_ranked_higher():
    high_rank = [make_recommendation("q1", ["i1", "i2"])]
    low_rank = [make_recommendation("q1", ["i2", "i1"])]
    relevant = {"q1": {"i1"}}
    assert ndcg_at_k(high_rank, relevant, k=2) > ndcg_at_k(low_rank, relevant, k=2)


def test_ndcg_at_k_on_empty_list_is_zero():
    assert ndcg_at_k([], {}, k=2) == 0.0
