from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationPoint:
    query_id: str
    doc_id: str
    score: float
    relevant: bool


def _metrics(points: list[CalibrationPoint], threshold: float) -> dict[str, float | int]:
    true_positive = sum(point.relevant and point.score >= threshold for point in points)
    false_positive = sum(not point.relevant and point.score >= threshold for point in points)
    false_negative = sum(point.relevant and point.score < threshold for point in points)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def select_threshold(
    points: list[CalibrationPoint], *, minimum_recall: float = 0.95
) -> dict[str, float | int]:
    if not points:
        raise ValueError("calibration points must not be empty")
    thresholds = sorted({0.0, 1.0, *(point.score for point in points)})
    candidates = [_metrics(points, threshold) for threshold in thresholds]
    feasible = [row for row in candidates if float(row["recall"]) >= minimum_recall]
    pool = feasible or candidates
    return max(
        pool,
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["recall"]),
            float(row["threshold"]),
        ),
    )


def query_gate_metrics(points: list[CalibrationPoint], threshold: float) -> dict[str, float | int]:
    grouped: dict[str, list[CalibrationPoint]] = {}
    for point in points:
        grouped.setdefault(point.query_id, []).append(point)
    no_relevant = [items for items in grouped.values() if not any(item.relevant for item in items)]
    answerable = [items for items in grouped.values() if any(item.relevant for item in items)]
    empty_correct = sum(not any(item.score >= threshold for item in items) for items in no_relevant)
    counterfactual_empty = [
        [item for item in items if not item.relevant]
        for items in grouped.values()
        if any(not item.relevant for item in items)
    ]
    counterfactual_empty_correct = sum(
        not any(item.score >= threshold for item in items) for items in counterfactual_empty
    )
    answer_found = sum(
        any(item.relevant and item.score >= threshold for item in items) for items in answerable
    )
    return {
        "queries": len(grouped),
        "answerable_queries": len(answerable),
        "no_relevant_candidate_queries": len(no_relevant),
        "answerable_recall": answer_found / max(1, len(answerable)),
        "empty_result_correctness": empty_correct / max(1, len(no_relevant)),
        "counterfactual_empty_queries": len(counterfactual_empty),
        "counterfactual_empty_result_correctness": counterfactual_empty_correct
        / max(1, len(counterfactual_empty)),
    }
