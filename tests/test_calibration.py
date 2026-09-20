from jev_rag_benchmark.calibration import (
    CalibrationPoint,
    query_gate_metrics,
    select_threshold,
)


def test_select_threshold_respects_minimum_recall():
    points = [
        CalibrationPoint("q1", "gold", 0.9, True),
        CalibrationPoint("q1", "noise", 0.6, False),
        CalibrationPoint("q2", "gold", 0.7, True),
        CalibrationPoint("q2", "noise", 0.2, False),
    ]
    result = select_threshold(points, minimum_recall=1.0)
    assert result["threshold"] == 0.7
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0


def test_query_gate_metrics_counts_empty_candidate_sets():
    points = [
        CalibrationPoint("answerable", "gold", 0.8, True),
        CalibrationPoint("empty", "noise", 0.1, False),
    ]
    result = query_gate_metrics(points, 0.5)
    assert result["answerable_recall"] == 1.0
    assert result["empty_result_correctness"] == 1.0
    assert result["counterfactual_empty_result_correctness"] == 1.0
