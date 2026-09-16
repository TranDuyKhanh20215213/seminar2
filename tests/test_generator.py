from src.generator import FakeGenerator
from src.models import Item, RetrievedDocument


def make_docs():
    return [
        RetrievedDocument(item=Item(item_id="i1", title="Mouse", description="Ergonomic mouse"), score=0.9),
        RetrievedDocument(item=Item(item_id="i2", title="Keyboard", description="Quiet keyboard"), score=0.8),
    ]


def test_fake_generator_mentions_all_context_titles():
    generator = FakeGenerator()
    output = generator.generate("office setup", make_docs())
    assert "Mouse" in output
    assert "Keyboard" in output


def test_fake_generator_records_last_query_text():
    generator = FakeGenerator()
    generator.generate("office setup", make_docs())
    assert generator.last_query_text == "office setup"
