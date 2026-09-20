from jev_rag_benchmark.stability import spearman, top_k_jaccard


def test_stability_metrics_detect_reversal():
    first = {"a": 1.0, "b": 0.5, "c": 0.1}
    same = {"a": 0.9, "b": 0.4, "c": 0.0}
    reverse = {"a": 0.1, "b": 0.5, "c": 1.0}
    assert spearman(first, same) == 1.0
    assert spearman(first, reverse) == -1.0
    assert top_k_jaccard(first, reverse, 1) == 0.0
