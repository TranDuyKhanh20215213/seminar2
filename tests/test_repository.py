# tests/test_repository.py
import pytest

from src.models import Item
from src.repository.memory_repository import InMemoryItemRepository
from src.repository.file_repository import FileItemRepository


@pytest.fixture(params=["memory", "file"])
def repository(request, tmp_path):
    if request.param == "memory":
        return InMemoryItemRepository()
    return FileItemRepository(str(tmp_path / "items.json"))


def test_add_then_get_returns_same_item(repository):
    repository.add(Item(item_id="i1", title="Mouse", description="Ergonomic mouse"))
    fetched = repository.get("i1")
    assert fetched.item_id == "i1"
    assert fetched.title == "Mouse"


def test_get_missing_item_returns_none(repository):
    assert repository.get("missing") is None


def test_all_returns_every_added_item(repository):
    repository.add(Item(item_id="i1", title="Mouse", description="d1"))
    repository.add(Item(item_id="i2", title="Keyboard", description="d2"))
    ids = {item.item_id for item in repository.all()}
    assert ids == {"i1", "i2"}


def test_remove_deletes_item(repository):
    repository.add(Item(item_id="i1", title="Mouse", description="d1"))
    repository.remove("i1")
    assert repository.get("i1") is None
    assert repository.all() == []


def test_file_repository_persists_across_new_instances(tmp_path):
    file_path = str(tmp_path / "items.json")
    first = FileItemRepository(file_path)
    first.add(Item(item_id="i1", title="Mouse", description="d1"))

    second = FileItemRepository(file_path)
    assert second.get("i1").title == "Mouse"


def test_file_repository_creates_parent_directory(tmp_path):
    file_path = str(tmp_path / "nested" / "dir" / "items.json")
    repository = FileItemRepository(file_path)
    repository.add(Item(item_id="i1", title="Mouse", description="d1"))
    assert repository.get("i1").title == "Mouse"
