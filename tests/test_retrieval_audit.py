import json
from pathlib import Path

from typer.testing import CliRunner

from jev_rag_benchmark import cli


FIXTURES = Path(__file__).parent / "fixtures"


def test_retrieval_audit_reports_candidate_ceiling(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_dataset_paths",
        lambda _: (FIXTURES / "documents.jsonl", FIXTURES / "queries.jsonl"),
    )
    result = CliRunner().invoke(
        cli.app,
        ["retrieval-audit", "--dataset", "xquad-tr", "--candidate-ks", "1,3,6"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["backend"] == "bm25"
    assert [row["k"] for row in payload["results"]] == [1, 3, 6]
    assert payload["results"][-1]["candidate_recall"] == 1.0
