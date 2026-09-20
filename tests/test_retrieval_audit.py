import json

from typer.testing import CliRunner

from jev_rag_benchmark.cli import app


def test_retrieval_audit_reports_candidate_ceiling():
    result = CliRunner().invoke(
        app,
        ["retrieval-audit", "--dataset", "xquad-tr", "--candidate-ks", "5,20,240"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["backend"] == "bm25"
    assert [row["k"] for row in payload["results"]] == [5, 20, 240]
    assert payload["results"][-1]["candidate_recall"] == 1.0
