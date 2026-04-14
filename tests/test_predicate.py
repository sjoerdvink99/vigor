import pandas as pd
import pytest
from src.predicate import Predicate


@pytest.fixture
def df():
    return pd.DataFrame({"density": [0.05, 0.15, 0.5], "n_nodes": [10, 50, 200]})


def test_fit_single_clause(df):
    p = Predicate(clauses={"density": [0, 0.1]})
    p.fit(df)
    assert p.mask.sum() == 1


def test_fit_conjunction(df):
    p = Predicate(clauses={"density": [0, 0.2], "n_nodes": [0, 100]})
    p.fit(df)
    assert p.mask.sum() == 2


def test_update_clause(df):
    p = Predicate(clauses={"density": [0, 0.1]})
    p.fit(df)
    p2 = p.update_clause("density", 0, 0.2)
    p2.fit(df)
    assert p2.mask.sum() == 2


def test_repr(df):
    p = Predicate(clauses={"density": [0, 0.1]})
    assert "density" in repr(p)


def test_from_clause_list():
    clause_list = [{"attribute": "density", "interval": [0, 0.1]}]
    p = Predicate(clause_list)
    assert "density" in p.clauses
