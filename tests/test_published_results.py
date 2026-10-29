import importlib.util
from pathlib import Path


def test_published_results_match_frozen_inputs_and_scoring():
    path = Path(__file__).resolve().parents[1] / "scripts/publish_results.py"
    spec = importlib.util.spec_from_file_location("published_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, summary = module.validate()
    assert len(rows) == summary["completed"] == 1044
