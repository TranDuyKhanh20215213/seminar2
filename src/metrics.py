from typing import List

from src.models import Recommendation


def attack_success_rate(recommendations: List[Recommendation], target_item_id: str) -> float:
    if not recommendations:
        return 0.0
    hits = sum(
        1 for rec in recommendations
        if any(item.item_id == target_item_id for item in rec.ranked_items)
    )
    return hits / len(recommendations)
