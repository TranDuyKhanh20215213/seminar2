from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.models import Query, RetrievedDocument


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
