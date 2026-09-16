import json
import os
from dataclasses import asdict
from typing import List, Optional

from src.models import Item
from src.repository.base import ItemRepository


class FileItemRepository(ItemRepository):
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump({}, f)

    def _load(self) -> dict:
        with open(self.file_path) as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, item: Item) -> None:
        data = self._load()
        data[item.item_id] = asdict(item)
        self._save(data)

    def get(self, item_id: str) -> Optional[Item]:
        raw = self._load().get(item_id)
        return Item(**raw) if raw is not None else None

    def all(self) -> List[Item]:
        return [Item(**raw) for raw in self._load().values()]

    def remove(self, item_id: str) -> None:
        data = self._load()
        data.pop(item_id, None)
        self._save(data)
