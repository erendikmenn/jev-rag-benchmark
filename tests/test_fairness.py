import json
import pytest
from jev_rag_benchmark.fairness import freeze_candidates, validate_comparison
from jev_rag_benchmark.models import Document, Query, Candidate


def test_snapshot_reuses_exact_text_and_order_without_retrieval(tmp_path):
    doc = Document("d", "evidence")
    query = Query("q", "question")

    class Index:
        def retrieve(self, query, k):
            return [Candidate(doc, 0.5, 1)]

    path = tmp_path / "pool.json"
    first, digest = freeze_candidates(path, [doc], [query], {"candidate_k": 1}, Index)

    def forbidden():
        raise AssertionError("retrieval must not run again")

    second, again = freeze_candidates(path, [doc], [query], {"candidate_k": 1}, forbidden)
    assert first == second and digest == again
    with pytest.raises(ValueError, match="settings"):
        freeze_candidates(path, [Document("d", "changed")], [query], {"candidate_k": 1}, Index)
    payload = json.loads(path.read_text())
    payload["queries"]["q"][0]["document"]["text"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="modified"):
        freeze_candidates(path, [doc], [query], {"candidate_k": 1}, Index)


def test_comparison_rejects_changed_candidates_protocol_and_coverage():
    a = dict(query_id="q", branch="A", protocol_id="locked", candidate_ids=["d", "e"])
    b = {**a, "branch": "D"}
    validate_comparison([a, b])
    for update in [
        {"candidate_ids": ["e", "d"]},
        {"protocol_id": "other"},
        {"query_id": "missing"},
    ]:
        with pytest.raises(ValueError):
            validate_comparison([a, {**b, **update}])
    with pytest.raises(ValueError):
        validate_comparison([a, a])


def test_benchmark_reuses_pool_and_scores_short_answer(tmp_path, monkeypatch):
    from pathlib import Path
    from jev_rag_benchmark import benchmark
    from jev_rag_benchmark.config import load_config
    from jev_rag_benchmark.generation import GenerationResult
    from jev_rag_benchmark.io import write_jsonl

    config = load_config(Path("configs/default.yaml"))
    config["retrieval"]["candidate_k"] = 1
    config["retrieval"]["context_k"] = 5
    docs, queries = tmp_path / "docs.jsonl", tmp_path / "queries.jsonl"
    write_jsonl(docs, [{"doc_id": "d", "text": "Panthers 308 sayı bıraktı."}])
    write_jsonl(
        queries,
        [
            {
                "query_id": "q",
                "text": "Kaç sayı?",
                "relevant_doc_ids": ["d"],
                "answers": ["308"],
                "language": "tr",
                "split": "test",
            }
        ],
    )

    class Generator:
        def generate(self, prompt):
            assert "shortest answer span" in prompt
            return GenerationResult("308 [d]", 1.0, "test")

    monkeypatch.setattr(benchmark, "create_generator", lambda *args: Generator())
    args = dict(
        documents_path=docs,
        queries_path=queries,
        output_path=tmp_path / "out.jsonl",
        config=config,
        branches=["A", "D"],
        split="test",
        limit=None,
        fixture_jev=True,
    )
    first = benchmark.run_benchmark(**args)

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot must avoid new retrieval")

    monkeypatch.setattr(benchmark, "create_retrieval_index", forbidden)
    second = benchmark.run_benchmark(**args)
    validate_comparison(second)
    assert all(r["answer_f1"] == r["answer_em"] == 1 for r in second)
    assert all(r["ndcg_5_context"] == 1 for r in second)
    assert first[0]["candidate_pool_sha256"] == second[0]["candidate_pool_sha256"]
