import gzip
import importlib.util
import json
from pathlib import Path


def test_luna_decisions_matches_prior_reranker_inputs():
    root = Path(__file__).resolve().parents[1]

    def load(name):
        spec = importlib.util.spec_from_file_location(name, root / f"scripts/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    luna, jev = load("run_luna_haiku"), load("run_jev114")
    pool = json.loads(
        gzip.decompress((root / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
    )
    for row in (pool[0], pool[-1]):
        for index in (None, 0, 19):
            actual = luna.body(row, "batch" if index is None else "pointwise", index, None)
            expected = jev.body(row, index)
            assert actual["state"] == expected["state"]
            assert actual["questions"] == expected["questions"]
