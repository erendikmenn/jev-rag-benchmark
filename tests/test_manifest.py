from datetime import date
import json

from jev_rag_benchmark.manifest import write_manifest


def test_manifest_serializes_yaml_dates(tmp_path):
    target = tmp_path / "manifest.json"
    write_manifest(target, {"verified_on": date(2026, 9, 19)})
    assert json.loads(target.read_text())["verified_on"] == "2026-09-19"
