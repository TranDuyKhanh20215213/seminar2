from typing import Dict, List, Optional

from src.models import Item
from src.repository.base import ItemRepository


class InMemoryItemRepository(ItemRepository):
    def __init__(self):
        self._items: Dict[str, Item] = {}

    def add(self, item: Item) -> None:
        self._items[item.item_id] = item

    def get(self, item_id: str) -> Optional[Item]:
        return self._items.get(item_id)

    def all(self) -> List[Item]:
        return list(self._items.values())

    def remove(self, item_id: str) -> None:
        self._items.pop(item_id, None)
