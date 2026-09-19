import httpx

from jev_rag_benchmark.models import Candidate, Document
from jev_rag_benchmark.rerankers import JevReranker


def candidates():
    return [
        Candidate(Document("d1", "Ankara is the capital of Türkiye."), 1.0, 1),
        Candidate(Document("d2", "Paris is the capital of France."), 0.5, 2),
    ]


def test_jev_parses_noul_and_records_resolved_model():
    def handler(request: httpx.Request):
        payload = __import__("json").loads(request.content)
        assert payload["model"] == "jev-1.13.0"
        assert payload["questions"]["candidate_0"]["type"] == "noul"
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "candidate_0": {"type": "noul", "noul": 0.95},
                    "candidate_1": {"type": "noul", "noul": 0.1},
                },
                "usage": {"input_tokens": 100, "output_tokens": 2},
            },
            headers={"x-typesafe-request-id": "req_test"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reranker = JevReranker(api_key="test", client=client, max_budget_usd=1)
    selected, telemetry = reranker.rerank("capital Türkiye", candidates(), 1)
    assert selected[0].document.doc_id == "d1"
    assert telemetry.resolved_model == "jev-1.13.0"
    assert telemetry.input_tokens == 100
    assert telemetry.request_id == "req_test"


def test_jev_falls_back_without_key():
    selected, telemetry = JevReranker(api_key=None).rerank("query", candidates(), 1)
    assert selected[0].document.doc_id == "d1"
    assert telemetry.fallback is True
    assert telemetry.estimated_cost_usd == 0

