# RAG Attack & Defense (AI4SE Topic 12 — Part 2/3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Poison-RAG-style adversarial data-poisoning attack against the Part 1 RAG recommender, and three retrieval-side countermeasures.

**Architecture:** An `Attacker` crafts adversarial `Item`s optimized (via embedding similarity) to be retrieved for a target set of user queries, and injects them through the same `ItemRepository` interface the rest of the system already uses — no special-casing needed anywhere else. `Defense` strategies plug into `RAGRecommender.defenses` (already wired in Part 1) to filter retrieved documents before generation.

**Tech Stack:** Python 3.10+, numpy, pytest. No new external dependencies beyond Part 1.

**Spec:** `AI4SE_Projects_HUST_2026.pdf`, Project 12 (pages 3-4). Reference paper: *Poison-RAG: Adversarial Data Poisoning Attacks on Retrieval-Augmented Generation in Recommender Systems*.

**Series:** Part 2 of 3 for Topic 12.
- **Part 1:** [`2026-09-16-rag-core-pipeline.md`](2026-09-16-rag-core-pipeline.md) — core RAG pipeline. **Must be implemented before this file.**
- **Part 2 (this file):** attacker + defenses.
- **Part 3:** [`2026-09-16-rag-evaluation-experiments.md`](2026-09-16-rag-evaluation-experiments.md) — metrics, experiment runner, notebook, report. Depends on this file.

## Global Constraints

(Same as Part 1 — repeated here since this file may be executed independently.)

- Data must be manageable both in memory and through files on disk, with minimal changes required to swap between them and no change to class/interface representation or business logic (spec p.1). Already satisfied by Part 1's `ItemRepository`; this part must keep using that interface rather than depending on a concrete backend.
- Regular GitHub commits (spec p.1) → one commit per task.
- PEP 8, functions well organized, comments only on non-obvious logic (spec p.1).
- Use open generative AI models: Llama, gpt-oss-20B, Qwen, Mistral (spec p.4) — already reachable via Part 1's `HFGenerator`; this part must not hardcode a specific model.
- Unit tests must never require downloading model weights → use `FakeEmbeddingModel` from Part 1 throughout.

---

## Prerequisites

Part 1 must already be implemented, providing: `src/models.py` (`Item`, `Query`, `RetrievedDocument`), `src/repository/base.py` (`ItemRepository`) and `src/repository/memory_repository.py` (`InMemoryItemRepository`), `src/embeddings.py` (`EmbeddingModel`, `FakeEmbeddingModel`), `src/rag_pipeline.py` (`RAGRecommender`, whose `defenses` parameter this part's `Defense` classes plug into).

## File Structure (Part 2 scope)

```
seminar2/
├── src/
│   ├── attacker.py
│   └── defenses.py
└── tests/
    ├── test_attacker.py
    └── test_defenses.py
```

---

### Task 2.1: Poison-RAG attacker

**Files:**
- Create: `src/attacker.py`
- Test: `tests/test_attacker.py`

**Interfaces:**
- Consumes: `EmbeddingModel` (Part 1, Task 1.5), `Item` (Part 1, Task 1.2), `ItemRepository` (Part 1, Task 1.3).
- Produces: `PoisonRAGAttacker(embedding_model: EmbeddingModel)` with `craft_adversarial_item(target_item_id: str, promotional_text: str, seed_queries: List[str], vocabulary: List[str], iterations: int = 20) -> Item` (returned item has `metadata["is_adversarial"] = True`, `metadata["target_item_id"]`, `metadata["attack_score"]`) and `inject(repository: ItemRepository, adversarial_items: List[Item]) -> None`. Part 3's `ExperimentRunner` calls both of these.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_attacker.py
from src.attacker import PoisonRAGAttacker
from src.embeddings import FakeEmbeddingModel
from src.models import Item
from src.repository.memory_repository import InMemoryItemRepository


def test_craft_adversarial_item_is_flagged_and_targets_given_item():
    attacker = PoisonRAGAttacker(FakeEmbeddingModel(dim=8))
    item = attacker.craft_adversarial_item(
        target_item_id="i1",
        promotional_text="Buy the best deal now",
        seed_queries=["quiet mechanical keyboard", "ergonomic keyboard for office"],
        vocabulary=["keyboard", "office", "quiet", "ergonomic", "deal"],
        iterations=5,
    )
    assert item.metadata["is_adversarial"] is True
    assert item.metadata["target_item_id"] == "i1"
    assert "Buy the best deal now" in item.description


def test_craft_adversarial_item_increases_similarity_to_seed_queries():
    attacker = PoisonRAGAttacker(FakeEmbeddingModel(dim=8))
    seed_queries = ["quiet mechanical keyboard", "ergonomic keyboard for office"]
    baseline_score = attacker._similarity(
        "Buy the best deal now", FakeEmbeddingModel(dim=8).encode(seed_queries).mean(axis=0)
    )
    item = attacker.craft_adversarial_item(
        target_item_id="i1",
        promotional_text="Buy the best deal now",
        seed_queries=seed_queries,
        vocabulary=["keyboard", "office", "quiet", "ergonomic", "deal"],
        iterations=20,
    )
    assert item.metadata["attack_score"] >= baseline_score


def test_inject_adds_adversarial_items_to_repository():
    attacker = PoisonRAGAttacker(FakeEmbeddingModel(dim=8))
    repository = InMemoryItemRepository()
    adversarial_item = Item(item_id="poison-i1", title="Sponsored", description="desc", metadata={"is_adversarial": True})

    attacker.inject(repository, [adversarial_item])

    assert repository.get("poison-i1") is not None
    assert repository.get("poison-i1").metadata["is_adversarial"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_attacker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.attacker'`

- [ ] **Step 3: Implement `src/attacker.py`**

```python
from typing import List

import numpy as np

from src.embeddings import EmbeddingModel
from src.models import Item
from src.repository.base import ItemRepository


class PoisonRAGAttacker:
    """Crafts adversarial catalog items whose embedding is optimized to be
    retrieved for a target set of user queries, following the Poison-RAG
    data-poisoning strategy: greedily append vocabulary terms that increase
    cosine similarity to the target query centroid while keeping the
    attacker-controlled promotional payload intact."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def craft_adversarial_item(
        self,
        target_item_id: str,
        promotional_text: str,
        seed_queries: List[str],
        vocabulary: List[str],
        iterations: int = 20,
    ) -> Item:
        target_vector = self.embedding_model.encode(seed_queries).mean(axis=0)

        best_text = promotional_text
        best_score = self._similarity(best_text, target_vector)

        rng = np.random.RandomState(0)
        for _ in range(iterations):
            candidate_word = vocabulary[rng.randint(len(vocabulary))]
            candidate_text = f"{best_text} {candidate_word}"
            candidate_score = self._similarity(candidate_text, target_vector)
            if candidate_score > best_score:
                best_text, best_score = candidate_text, candidate_score

        return Item(
            item_id=f"poison-{target_item_id}",
            title=f"Sponsored: {target_item_id}",
            description=best_text,
            metadata={
                "is_adversarial": True,
                "target_item_id": target_item_id,
                "attack_score": best_score,
            },
        )

    def _similarity(self, text: str, target_vector: np.ndarray) -> float:
        vector = self.embedding_model.encode([text])[0]
        denom = (np.linalg.norm(vector) * np.linalg.norm(target_vector)) + 1e-10
        return float(vector @ target_vector / denom)

    def inject(self, repository: ItemRepository, adversarial_items: List[Item]) -> None:
        for item in adversarial_items:
            repository.add(item)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_attacker.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/attacker.py tests/test_attacker.py
git commit -m "feat: add Poison-RAG adversarial item crafting and injection"
```

---

### Task 2.2: Outlier-filter defense

**Files:**
- Create: `src/defenses.py`
- Test: `tests/test_defenses.py`

**Interfaces:**
- Consumes: `RetrievedDocument`, `Query` (Part 1, Task 1.2).
- Produces: `Defense` ABC with `filter(query: Query, retrieved: List[RetrievedDocument]) -> List[RetrievedDocument]`; `OutlierFilterDefense(max_z_score: float = 2.0)`. Both `RAGRecommender.defenses` (Part 1) and Part 3's `ExperimentRunner` expect this exact `filter` signature.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_defenses.py
from src.defenses import OutlierFilterDefense
from src.models import Item, Query, RetrievedDocument


def make_query():
    return Query(query_id="q1", text="keyboard")


def test_outlier_filter_keeps_scores_within_z_threshold():
    docs = [
        RetrievedDocument(item=Item(item_id="i1", title="A", description="d"), score=0.9),
        RetrievedDocument(item=Item(item_id="i2", title="B", description="d"), score=0.88),
        RetrievedDocument(item=Item(item_id="i3", title="C", description="d"), score=0.05),
    ]
    defense = OutlierFilterDefense(max_z_score=1.0)
    filtered_ids = {doc.item.item_id for doc in defense.filter(make_query(), docs)}
    assert "i3" not in filtered_ids


def test_outlier_filter_passes_through_short_lists_unchanged():
    docs = [RetrievedDocument(item=Item(item_id="i1", title="A", description="d"), score=0.9)]
    defense = OutlierFilterDefense()
    assert defense.filter(make_query(), docs) == docs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_defenses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.defenses'`

- [ ] **Step 3: Implement `src/defenses.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_defenses.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/defenses.py tests/test_defenses.py
git commit -m "feat: add defense interface and outlier-filter defense"
```

---

### Task 2.3: Metadata-trust defense (oracle upper bound)

**Files:**
- Modify: `src/defenses.py` (add `MetadataTrustDefense`)
- Modify: `tests/test_defenses.py` (add test)

**Interfaces:**
- Consumes: `Defense` (Task 2.2), `Item.metadata` flag set by `PoisonRAGAttacker` (Task 2.1).
- Produces: `MetadataTrustDefense(flag_key: str = "is_adversarial")`.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_defenses.py`:

```python
from src.defenses import MetadataTrustDefense


def test_metadata_trust_defense_removes_flagged_items():
    docs = [
        RetrievedDocument(item=Item(item_id="i1", title="A", description="d"), score=0.9),
        RetrievedDocument(item=Item(item_id="poison-i1", title="B", description="d", metadata={"is_adversarial": True}), score=0.95),
    ]
    defense = MetadataTrustDefense()
    filtered_ids = {doc.item.item_id for doc in defense.filter(make_query(), docs)}
    assert filtered_ids == {"i1"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_defenses.py -v`
Expected: FAIL with `ImportError: cannot import name 'MetadataTrustDefense'`

- [ ] **Step 3: Add the implementation**

Append to `src/defenses.py`:

```python
class MetadataTrustDefense(Defense):
    """Removes items flagged as adversarial. This is an oracle upper bound
    used to measure the best possible defense outcome in experiments — a
    real deployment cannot see the attacker's intent flag directly."""

    def __init__(self, flag_key: str = "is_adversarial"):
        self.flag_key = flag_key

    def filter(self, query: Query, retrieved: List[RetrievedDocument]) -> List[RetrievedDocument]:
        return [doc for doc in retrieved if not doc.item.metadata.get(self.flag_key, False)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_defenses.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/defenses.py tests/test_defenses.py
git commit -m "feat: add metadata-trust oracle defense as an evaluation upper bound"
```

---

### Task 2.4: Majority-agreement defense

**Files:**
- Modify: `src/defenses.py` (add `MajorityAgreementDefense`)
- Modify: `tests/test_defenses.py` (add test)

**Interfaces:**
- Consumes: `Defense` (Task 2.2), `EmbeddingModel`/`FakeEmbeddingModel` (Part 1, Task 1.5).
- Produces: `MajorityAgreementDefense(embedding_model: EmbeddingModel, min_neighbors: int = 2, similarity_threshold: float = 0.5)`. This completes the set of three `Defense` implementations Part 3's `ExperimentRunner` compares.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_defenses.py`:

```python
from src.defenses import MajorityAgreementDefense
from src.embeddings import FakeEmbeddingModel


def test_majority_agreement_defense_drops_isolated_outlier_text():
    docs = [
        RetrievedDocument(item=Item(item_id="i1", title="Mechanical Keyboard", description="Quiet mechanical keyboard"), score=0.9),
        RetrievedDocument(item=Item(item_id="i2", title="Mechanical Keyboard", description="Quiet mechanical keyboard"), score=0.89),
        RetrievedDocument(item=Item(item_id="i3", title="Mechanical Keyboard", description="Quiet mechanical keyboard"), score=0.88),
        RetrievedDocument(item=Item(item_id="poison", title="Totally unrelated payload", description="Buy now discount offer"), score=0.95),
    ]
    defense = MajorityAgreementDefense(FakeEmbeddingModel(dim=8), min_neighbors=2, similarity_threshold=0.9)
    kept_ids = {doc.item.item_id for doc in defense.filter(make_query(), docs)}
    assert "poison" not in kept_ids
    assert {"i1", "i2", "i3"}.issubset(kept_ids)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_defenses.py -v`
Expected: FAIL with `ImportError: cannot import name 'MajorityAgreementDefense'`

- [ ] **Step 3: Add the implementation**

Append to `src/defenses.py`:

```python
from src.embeddings import EmbeddingModel


class MajorityAgreementDefense(Defense):
    """Keeps a retrieved document only if enough other retrieved documents
    are semantically similar to it — an adversarial document crafted to
    match the query embedding but unrelated in content to genuine catalog
    items tends to be an outlier in this cross-document similarity graph."""

    def __init__(self, embedding_model: EmbeddingModel, min_neighbors: int = 2, similarity_threshold: float = 0.5):
        self.embedding_model = embedding_model
        self.min_neighbors = min_neighbors
        self.similarity_threshold = similarity_threshold

    def filter(self, query: Query, retrieved: List[RetrievedDocument]) -> List[RetrievedDocument]:
        if len(retrieved) <= self.min_neighbors:
            return retrieved
        texts = [f"{doc.item.title}. {doc.item.description}" for doc in retrieved]
        vectors = self.embedding_model.encode(texts)
        normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        similarity_matrix = normalized @ normalized.T

        kept = []
        for i, doc in enumerate(retrieved):
            neighbor_count = np.sum(similarity_matrix[i] >= self.similarity_threshold) - 1
            if neighbor_count >= self.min_neighbors:
                kept.append(doc)
        return kept if kept else retrieved
```

Note: place the `from src.embeddings import EmbeddingModel` import at the top of `src/defenses.py` alongside the existing imports, not inline.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_defenses.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the combined Part 1 + Part 2 test suite**

Run: `pytest -v`
Expected: all tests from Part 1 and Part 2 PASS (integration tests deselected by default).

- [ ] **Step 6: Commit**

```bash
git add src/defenses.py tests/test_defenses.py
git commit -m "feat: add majority-agreement defense"
```

---

## Self-Review Notes (Part 2)

- **Coverage:** adversarial attack (Poison-RAG) → Task 2.1; countermeasures including the ICL/prompt-engineering/CoT technique named in the spec's "Tasks" bullet → Task 2.2-2.4 (retrieval-side) plus Part 1's `PromptGuardGenerator` (generation-side, already built).
- **Type consistency:** `Defense.filter(query, retrieved)` signature identical across `OutlierFilterDefense`, `MetadataTrustDefense`, `MajorityAgreementDefense`, and matches what `RAGRecommender.defenses` (Part 1) and `ExperimentRunner` (Part 3) call.
- **Handoff to Part 3:** Part 3 imports `PoisonRAGAttacker` (Task 2.1) and all three `Defense` classes (Tasks 2.2-2.4) to build `ExperimentConfig`/`ExperimentRunner`.
