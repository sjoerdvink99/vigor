import numpy as np
import torch
import pytest
from src.model import PredicateInduction


def test_predict_output_range():
    model = PredicateInduction()
    x = torch.randn(20, 3)
    a = torch.ones(3)
    mu = torch.zeros(3)
    pred = model.predict(x, a, mu)
    assert pred.shape == (20,)
    assert (pred >= 0).all() and (pred <= 1).all()


def test_predict_center_is_max():
    model = PredicateInduction()
    a = torch.ones(2)
    mu = torch.tensor([0.5, 0.5])
    points = torch.tensor([[0.5, 0.5], [0.9, 0.9]])
    preds = model.predict(points, a, mu)
    assert preds[0] > preds[1]


def test_compute_predicate_sequence_recovers_interval():
    np.random.seed(0)
    X = np.random.uniform(0, 1, (200, 2)).astype(np.float32)
    selected = ((X[:, 0] > 0.3) & (X[:, 0] < 0.7)).astype(bool)

    model = PredicateInduction()
    result = model.compute_predicate_sequence(
        X, np.array([selected]), attribute_names=["a", "b"], n_iter=500
    )
    assert len(result) == 1


def test_compute_predicate_sequence_returns_clauses():
    np.random.seed(1)
    X = np.random.uniform(0, 1, (100, 3)).astype(np.float32)
    selected = (X[:, 0] > 0.5).astype(bool)

    model = PredicateInduction()
    result = model.compute_predicate_sequence(
        X, np.array([selected]), attribute_names=["x", "y", "z"], n_iter=200
    )
    assert len(result) == 1
    for clause in result[0]:
        assert "attribute" in clause
        assert "interval" in clause
        assert clause["interval"][0] < clause["interval"][1]


def test_compute_predicate_sequence_with_prior():
    np.random.seed(2)
    X = np.random.uniform(0, 1, (200, 2)).astype(np.float32)
    selected = ((X[:, 0] > 0.3) & (X[:, 0] < 0.7)).astype(bool)

    priors = [{"a": (0.3, 0.7)}]

    model = PredicateInduction()
    result = model.compute_predicate_sequence(
        X, np.array([selected]), attribute_names=["a", "b"], n_iter=300, priors=priors
    )
    assert len(result) == 1
    attrs = {clause["attribute"] for clause in result[0]}
    assert "a" in attrs
