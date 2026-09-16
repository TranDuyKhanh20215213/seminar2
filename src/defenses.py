from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.embeddings import EmbeddingModel
from src.models import Query, RetrievedDocument, item_text


class Defense(ABC):
    @abstractmethod
    def filter(self, query: Query, retrieved: List[RetrievedDocument]) -> List[RetrievedDocument]:
        ...


class OutlierFilterDefense(Defense):
    """Drops retrieved documents whose similarity score is an outlier
    relative to the rest of the retrieved set — adversarial items crafted to
    maximize similarity to a narrow set of seed queries often score
    anomalously high or, on off-target queries, anomalously low."""

    def __init__(self, max_z_score: float = 2.0):
        self.max_z_score = max_z_score

    def filter(self, query: Query, retrieved: List[RetrievedDocument]) -> List[RetrievedDocument]:
        if len(retrieved) < 3:
            return retrieved
        scores = np.array([doc.score for doc in retrieved])
        mean, std = scores.mean(), scores.std() + 1e-10
        z_scores = np.abs((scores - mean) / std)
        return [doc for doc, z in zip(retrieved, z_scores) if z <= self.max_z_score]


class MetadataTrustDefense(Defense):
    """Removes items flagged as adversarial. This is an oracle upper bound
    used to measure the best possible defense outcome in experiments — a
    real deployment cannot see the attacker's intent flag directly."""

    def __init__(self, flag_key: str = "is_adversarial"):
        self.flag_key = flag_key

    def filter(self, query: Query, retrieved: List[RetrievedDocument]) -> List[RetrievedDocument]:
        return [doc for doc in retrieved if not doc.item.metadata.get(self.flag_key, False)]


class MajorityAgreementDefense(Defense):
    """Keeps a retrieved document only if enough other retrieved documents
    are semantically similar to it — an adversarial document crafted to
    match the query embedding but unrelated in content to genuine catalog
    items tends to be an outlier in this cross-document similarity graph.
    If nothing clears the neighbor threshold, this defense fails open and
    returns the unfiltered list rather than an empty one — it declines to
    act under uncertainty instead of blocking everything."""

    def __init__(self, embedding_model: EmbeddingModel, min_neighbors: int = 2, similarity_threshold: float = 0.5):
        self.embedding_model = embedding_model
        self.min_neighbors = min_neighbors
        self.similarity_threshold = similarity_threshold

    def filter(self, query: Query, retrieved: List[RetrievedDocument]) -> List[RetrievedDocument]:
        if len(retrieved) <= self.min_neighbors:
            return retrieved
        texts = [item_text(doc.item) for doc in retrieved]
        vectors = self.embedding_model.encode(texts)
        normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        similarity_matrix = normalized @ normalized.T

        kept = []
        for i, doc in enumerate(retrieved):
            neighbor_count = np.sum(similarity_matrix[i] >= self.similarity_threshold) - 1
            if neighbor_count >= self.min_neighbors:
                kept.append(doc)
        return kept if kept else retrieved
