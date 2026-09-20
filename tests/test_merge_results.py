from typer.testing import CliRunner

from jev_rag_benchmark.cli import app
from jev_rag_benchmark.io import read_jsonl, write_jsonl


def test_merge_results_sorts_and_rejects_no_rows(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    write_jsonl(first, [{"query_id": "q2", "branch": "D", "context_ids": ["d2"]}])
    write_jsonl(second, [{"query_id": "q1", "branch": "D", "context_ids": ["d1"]}])

    result = CliRunner().invoke(app, ["merge-results", str(output), f"{first},{second}"])

    assert result.exit_code == 0, result.output
    assert [row["query_id"] for row in read_jsonl(output)] == ["q1", "q2"]
    assert output.with_suffix(".csv").exists()
    assert output.with_suffix(".manifest.json").exists()
