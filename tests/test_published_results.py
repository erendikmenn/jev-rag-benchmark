import importlib.util
from pathlib import Path


def test_published_results_match_frozen_inputs_and_scoring():
    path = Path(__file__).resolve().parents[1] / "scripts/publish_results.py"
    spec = importlib.util.spec_from_file_location("published_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, summary = module.validate()
    assert len(rows) == summary["completed"] == 1044


def test_clef_flash_probability_ranking_and_frozen_pool():
    path = Path(__file__).resolve().parents[1] / "scripts/publish_results.py"
    spec = importlib.util.spec_from_file_location("published_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.validate_clef()
    import json

    saved = json.loads((module.ROOT / "results/clef-flash-2026-10-29/summary.json").read_text())
    for key, value in result.items():
        assert abs(saved[key] - value) < 1e-12
    assert 0 <= result["recall_at_5"] <= 1
    assert 0 < result["cost_usd"] < 0.60


def test_clef_requests_preserve_jev_batch_questions_and_candidate_texts():
    import gzip
    import json

    from jev_rag_benchmark.fairness import fingerprint

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("clef_runner", root / "scripts/run_clef_flash.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pool = json.loads(
        gzip.decompress((root / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
    )
    frozen = {r["query_id"]: r for r in pool}
    results = root / "results/clef-flash-2026-10-29/results.jsonl"
    for row in map(json.loads, results.read_text().splitlines()):
        body = module.request_body(frozen[row["query_id"]])
        assert fingerprint(body) == row["request_sha256"]
        assert body["provider"] == {"only": ["DeepInfra"], "allow_fallbacks": False}
        assert len(body["questions"]) == len(body["state"]["candidates"]) == 20
