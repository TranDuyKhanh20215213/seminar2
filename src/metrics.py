import math
from typing import Dict, List, Set

from src.models import Recommendation


def attack_success_rate(recommendations: List[Recommendation], target_item_id: str) -> float:
    """Fraction of recommendations that surface the target item anywhere in the ranked list."""
    if not recommendations:
        return 0.0
    hits = sum(
        1 for rec in recommendations
        if any(item.item_id == target_item_id for item in rec.ranked_items)
    )
    return hits / len(recommendations)


def hit_rate_at_k(recommendations: List[Recommendation], relevant_item_ids: Dict[str, Set[str]], k: int) -> float:
    """Fraction of recommendations whose top-k contains at least one relevant item."""
    if not recommendations:
        return 0.0
    hits = 0
    for rec in recommendations:
        relevant = relevant_item_ids.get(rec.query.query_id, set())
        top_k_ids = {item.item_id for item in rec.ranked_items[:k]}
        if top_k_ids & relevant:
            hits += 1
    return hits / len(recommendations)


def ndcg_at_k(recommendations: List[Recommendation], relevant_item_ids: Dict[str, Set[str]], k: int) -> float:
    """Mean normalized discounted cumulative gain at k, using binary relevance."""
    if not recommendations:
        return 0.0
    total = 0.0
    for rec in recommendations:
        relevant = relevant_item_ids.get(rec.query.query_id, set())
        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, item in enumerate(rec.ranked_items[:k])
            if item.item_id in relevant
        )
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        total += (dcg / idcg) if idcg > 0 else 0.0
    return total / len(recommendations)
