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


class HFGenerator(Generator):
    def __init__(self, model_name: str):
        from transformers import pipeline

        self._pipe = pipeline("text-generation", model=model_name)

    def generate(self, query_text: str, context_docs: List[RetrievedDocument]) -> str:
        context_text = "\n".join(f"- {doc.item.title}: {doc.item.description}" for doc in context_docs)
        prompt = (
            "You are a recommender assistant. Given the user's request and the "
            "retrieved candidate items below, recommend the most relevant items "
            "and briefly justify your choice.\n\n"
            f"User request: {query_text}\n\nCandidate items:\n{context_text}\n\nRecommendation:"
        )
        output = self._pipe(prompt, max_new_tokens=150, do_sample=False)
        return output[0]["generated_text"][len(prompt):].strip()
