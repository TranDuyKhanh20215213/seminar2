from abc import ABC, abstractmethod
from typing import List, Optional

from src.models import Item


class ItemRepository(ABC):
    @abstractmethod
    def add(self, item: Item) -> None:
        ...

    @abstractmethod
    def get(self, item_id: str) -> Optional[Item]:
        ...

    @abstractmethod
    def all(self) -> List[Item]:
        ...

    @abstractmethod
    def remove(self, item_id: str) -> None:
        ...
