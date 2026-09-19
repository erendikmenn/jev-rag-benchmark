from jev_rag_benchmark.metrics import (
    exact_match,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_ci,
    recall_at_k,
    token_f1,
)


def test_retrieval_metrics_reward_relevant_first():
    ranked = ["relevant", "other"]
    relevant = {"relevant"}
    assert ndcg_at_k(ranked, relevant) == 1.0
    assert mrr_at_k(ranked, relevant) == 1.0
    assert recall_at_k(ranked, relevant, 2) == 1.0


def test_answer_metrics_ignore_citations():
    assert exact_match("Ankara [d1]", ["Ankara"]) == 1.0
    assert token_f1("The Red Planet is Mars [d6]", ["Mars"]) > 0


def test_paired_bootstrap_is_deterministic():
    one = paired_bootstrap_ci([0, 0, 1], [1, 0, 1], seed=7, samples=2000)
    two = paired_bootstrap_ci([0, 0, 1], [1, 0, 1], seed=7, samples=2000)
    assert one == two

