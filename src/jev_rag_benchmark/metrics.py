from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from statistics import mean


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    actual = [1 if doc_id in relevant_ids else 0 for doc_id in ranked_ids[:k]]
    ideal = [1] * min(k, len(relevant_ids))
    denominator = dcg(ideal)
    return dcg(actual) / denominator if denominator else 0.0


def mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    for idx, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / idx
    return 0.0


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def normalize_answer(text: str) -> str:
    text = text.casefold()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, references: tuple[str, ...] | list[str]) -> float:
    normalized = normalize_answer(re.sub(r"\[[^\]]+\]", "", prediction))
    return float(any(normalized == normalize_answer(reference) for reference in references))


def token_f1(prediction: str, references: tuple[str, ...] | list[str]) -> float:
    pred = Counter(normalize_answer(re.sub(r"\[[^\]]+\]", "", prediction)).split())
    best = 0.0
    for reference in references:
        gold = Counter(normalize_answer(reference).split())
        overlap = sum((pred & gold).values())
        if not pred or not gold:
            score = float(pred == gold)
        else:
            precision = overlap / sum(pred.values())
            recall = overlap / sum(gold.values())
            score = 2 * precision * recall / (precision + recall) if overlap else 0.0
        best = max(best, score)
    return best


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def paired_bootstrap_ci(
    baseline: list[float], treatment: list[float], *, seed: int, samples: int = 2000
) -> tuple[float, float, float]:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired samples must be non-empty and equal length")
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        indexes = [rng.randrange(len(baseline)) for _ in baseline]
        deltas.append(mean(treatment[i] - baseline[i] for i in indexes))
    deltas.sort()
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return (
        mean(t - b for b, t in zip(baseline, treatment)),
        deltas[low_index],
        deltas[high_index],
    )
