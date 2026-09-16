# Topic 12 — Adversarial Attacks on RAG-based Recommender Systems

Implements a RAG recommender, a Poison-RAG-style data poisoning attack, and
defenses (retrieval-side filtering + prompt-engineering/Chain-of-Thought),
following AI4SE Project 12.

## Setup

    pip install -r requirements.txt

## Run unit tests (fast, no model downloads)

    pytest

## Run integration tests (downloads real embedding/LLM models)

    pytest -m integration

## Run the full experiment suite

    jupyter lab notebooks/run_experiments.ipynb

Results are written to `data/experiment_results.json`. See `report/report.tex`
for the write-up template.
