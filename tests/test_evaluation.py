import importlib

evaluation = importlib.import_module("07_evaluation")


def test_recall_at_k():
    assert evaluation.recall_at_k([1, 2, 3, 4], {2, 3}, k=3) == 1.0
    assert evaluation.recall_at_k([1, 2, 3, 4], {2, 3}, k=1) == 0.0
    assert evaluation.recall_at_k([1, 2], set(), 2) == 0.0


def test_mrr():
    assert evaluation.mrr([9, 1], {9}) == 1.0
    assert evaluation.mrr([5, 2, 9], {9}) == 1 / 3
    assert evaluation.mrr([1, 2], {9}) == 0.0


def test_run_experiment_returns_valid_metrics():
    row = evaluation.run_experiment(80, "tfidf")
    assert 0.0 <= row["recall"] <= 1.0
    assert 0.0 <= row["mrr"] <= 1.0
