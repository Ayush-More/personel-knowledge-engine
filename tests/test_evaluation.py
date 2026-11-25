
import pytest

from knowledge_engine.evaluation import precision_at_k , recall_at_k , reciprocal_rank


def test_precision_at_k():
    retrieved = ["x", "a", "y", "b", "z"]
    relevant = {"a", "b", "c"}

    assert precision_at_k(retrieved, relevant, 5) == pytest.approx(0.4)


def test_recall_at_k():
    retrieved = ["x", "a", "y", "b", "z"]
    relevant = {"a", "b", "c"}

    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    retrieved = ["x", "a", "y"]
    relevant = {"a", "b"}

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)


def test_reciprocal_rank_is_zero_when_nothing_matches():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_recall_rejects_empty_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k(["a", "b"], set(), 2)


def test_metrics_reject_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)
