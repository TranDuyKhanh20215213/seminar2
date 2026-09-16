import os
import subprocess
import sys

import numpy as np
import pytest

from src.embeddings import FakeEmbeddingModel, SentenceTransformerEmbedding


def test_fake_embedding_returns_one_vector_per_text():
    model = FakeEmbeddingModel(dim=8)
    vectors = model.encode(["hello world", "goodbye world"])
    assert vectors.shape == (2, 8)


def test_fake_embedding_is_deterministic_for_same_text():
    model = FakeEmbeddingModel(dim=8)
    first = model.encode(["same text"])[0]
    second = model.encode(["same text"])[0]
    assert np.allclose(first, second)


def test_fake_embedding_differs_for_different_text():
    model = FakeEmbeddingModel(dim=8)
    a = model.encode(["cats"])[0]
    b = model.encode(["airplanes"])[0]
    assert not np.allclose(a, b)


def test_fake_embedding_is_deterministic_across_processes():
    script = (
        "from src.embeddings import FakeEmbeddingModel\n"
        "model = FakeEmbeddingModel(dim=8)\n"
        "print(list(model.encode(['same text'])[0]))\n"
    )
    env_a = {**os.environ, "PYTHONHASHSEED": "1"}
    env_b = {**os.environ, "PYTHONHASHSEED": "2"}
    result_a = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=".", env=env_a)
    result_b = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=".", env=env_b)
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr
    assert result_a.stdout == result_b.stdout


@pytest.mark.integration
def test_sentence_transformer_embedding_returns_correct_shape():
    model = SentenceTransformerEmbedding()
    vectors = model.encode(["a quiet mechanical keyboard", "a wireless mouse"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0
