import importlib.util
from pathlib import Path


def test_rate_limit_reset_timestamp_is_milliseconds():
    path = Path(__file__).resolve().parents[1] / "scripts/run_jev114.py"
    spec = importlib.util.spec_from_file_location("jev114_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = {"error": {"metadata": {"headers": {"X-RateLimit-Reset": "1060000"}}}}
    assert module.rate_limit_delay(payload, 1000) == 61
    assert module.rate_limit_delay(payload, 1100) == 1
    assert module.rate_limit_delay({}, 1000) == 5


def test_jev114_reuses_prior_batch_and_pointwise_prompts():
    import gzip
    import json

    root = Path(__file__).resolve().parents[1]

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / f"scripts/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    jev = load("run_jev114")
    batch = load("run_clef_flash")
    pointwise = load("run_clef_pointwise")
    pool = json.loads(
        gzip.decompress((root / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
    )
    for row in (pool[0], pool[-1]):
        for index in (None, 0, 19):
            actual = jev.body(row, index)
            expected = batch.request_body(row) if index is None else pointwise.body(row, index)
            assert actual["state"] == expected["state"]
            assert actual["questions"] == expected["questions"]
            assert actual["model"] == "typesafe/jev-1.14"
