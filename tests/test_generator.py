import pytest

from src.generator import FakeGenerator, HFGenerator, PromptGuardGenerator
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


@pytest.mark.integration
def test_hf_generator_returns_nonempty_string():
    generator = HFGenerator(model_name="sshleifer/tiny-gpt2")
    output = generator.generate("office setup", make_docs())
    assert isinstance(output, str)
    assert len(output) > 0


def test_prompt_guard_generator_rewrites_query_with_cot_instruction():
    fake = FakeGenerator()
    guarded = PromptGuardGenerator(fake)
    guarded.generate("office setup", make_docs())
    assert "office setup" in fake.last_query_text
    assert "suspect" in fake.last_query_text.lower()


def test_prompt_guard_generator_still_returns_base_generator_output():
    fake = FakeGenerator()
    guarded = PromptGuardGenerator(fake)
    output = guarded.generate("office setup", make_docs())
    assert "Mouse" in output
