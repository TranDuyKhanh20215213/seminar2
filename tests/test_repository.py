from src.models import Item
from src.repository.memory_repository import InMemoryItemRepository


def make_repository():
    return InMemoryItemRepository()


def test_add_then_get_returns_same_item():
    repository = make_repository()
    repository.add(Item(item_id="i1", title="Mouse", description="Ergonomic mouse"))
    fetched = repository.get("i1")
    assert fetched.item_id == "i1"
    assert fetched.title == "Mouse"


def test_get_missing_item_returns_none():
    assert make_repository().get("missing") is None


def test_all_returns_every_added_item():
    repository = make_repository()
    repository.add(Item(item_id="i1", title="Mouse", description="d1"))
    repository.add(Item(item_id="i2", title="Keyboard", description="d2"))
    ids = {item.item_id for item in repository.all()}
    assert ids == {"i1", "i2"}


def test_remove_deletes_item():
    repository = make_repository()
    repository.add(Item(item_id="i1", title="Mouse", description="d1"))
    repository.remove("i1")
    assert repository.get("i1") is None
    assert repository.all() == []
