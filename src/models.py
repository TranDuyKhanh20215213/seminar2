from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Item:
    item_id: str
    title: str
    description: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Query:
    query_id: str
    text: str
    user_id: Optional[str] = None


@dataclass
class RetrievedDocument:
    item: Item
    score: float


@dataclass
class Recommendation:
    query: Query
    ranked_items: List[Item]
    explanation: str


def item_text(item: Item) -> str:
    return f"{item.title}. {item.description}"
