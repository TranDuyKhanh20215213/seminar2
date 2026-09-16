import numpy as np

from src.embeddings import FakeEmbeddingModel


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
