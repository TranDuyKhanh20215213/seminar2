from src.metrics import attack_success_rate
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
