from __future__ import annotations


def rank_map(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return {key: rank for rank, key in enumerate(ordered, start=1)}


def spearman(scores_a: dict[str, float], scores_b: dict[str, float]) -> float:
    shared = set(scores_a) & set(scores_b)
    count = len(shared)
    if count < 2:
        return 1.0
    ranks_a = rank_map({key: scores_a[key] for key in shared})
    ranks_b = rank_map({key: scores_b[key] for key in shared})
    squared = sum((ranks_a[key] - ranks_b[key]) ** 2 for key in shared)
    return 1 - (6 * squared) / (count * (count * count - 1))


def top_k_jaccard(scores_a: dict[str, float], scores_b: dict[str, float], k: int) -> float:
    top_a = set(sorted(scores_a, key=lambda key: (-scores_a[key], key))[:k])
    top_b = set(sorted(scores_b, key=lambda key: (-scores_b[key], key))[:k])
    union = top_a | top_b
    return len(top_a & top_b) / len(union) if union else 1.0
