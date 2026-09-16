from abc import ABC, abstractmethod
from typing import List

from src.models import RetrievedDocument


class Generator(ABC):
    @abstractmethod
    def generate(self, query_text: str, context_docs: List[RetrievedDocument]) -> str:
        ...


class FakeGenerator(Generator):
    """Deterministic stand-in for an LLM. Records the last prompt it saw so
    defenses that rewrite the query (e.g. PromptGuardGenerator) can be tested."""

    def __init__(self):
        self.last_query_text = ""

    def generate(self, query_text: str, context_docs: List[RetrievedDocument]) -> str:
        self.last_query_text = query_text
        titles = ", ".join(doc.item.title for doc in context_docs)
        return f"Recommended based on: {titles}"
