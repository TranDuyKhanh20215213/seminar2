# AI4SE Topic 12 — Plan Index

This feature was split into three smaller, sequential plan files instead of one large document. Execute them in order:

1. [`2026-09-16-rag-core-pipeline.md`](2026-09-16-rag-core-pipeline.md) — Part 1/3: domain models, swappable in-memory/file repository, embeddings, vector retriever, generator (fake/HF/prompt-guard), RAG pipeline. Produces a working RAG recommender with no attack yet. (Tasks 1.1–1.11)
2. [`2026-09-16-rag-attack-defense.md`](2026-09-16-rag-attack-defense.md) — Part 2/3: Poison-RAG adversarial attacker, plus outlier-filter / metadata-trust / majority-agreement defenses. Depends on Part 1. (Tasks 2.1–2.4)
3. [`2026-09-16-rag-evaluation-experiments.md`](2026-09-16-rag-evaluation-experiments.md) — Part 3/3: evaluation metrics, experiment runner, Jupyter notebook entry point, LaTeX report skeleton. Depends on Parts 1–2. (Tasks 3.1–3.6)

Each file carries its own copy of the shared Global Constraints (from `AI4SE_Projects_HUST_2026.pdf`, Project 12) and can be handed to `superpowers:subagent-driven-development` or `superpowers:executing-plans` independently, in the order above.
