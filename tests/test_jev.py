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
        assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
        assert payload["model"] == "typesafe/jev-1.13"
        assert payload["questions"]["candidate_0"]["type"] == "noul"
        return httpx.Response(
            200,
            json={
                "id": "gen-dec-test",
                "model": "typesafe/jev-1.13-20260917",
                "answers": {
                    "candidate_0": {"type": "noul", "noul": 0.95},
                    "candidate_1": {"type": "noul", "noul": 0.1},
                },
                "usage": {"input_tokens": 100, "output_tokens": 2, "cost": 0.0000042},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reranker = JevReranker(api_key="test", client=client, max_budget_usd=1)
    selected, telemetry = reranker.rerank("capital Türkiye", candidates(), 1)
    assert selected[0].document.doc_id == "d1"
    assert telemetry.resolved_model == "typesafe/jev-1.13-20260917"
    assert telemetry.input_tokens == 100
    assert telemetry.estimated_cost_usd == 0.0000042
    assert telemetry.request_id == "gen-dec-test"


def test_jev_falls_back_without_key():
    selected, telemetry = JevReranker(api_key=None).rerank("query", candidates(), 1)
    assert selected[0].document.doc_id == "d1"
    assert telemetry.fallback is True
    assert telemetry.estimated_cost_usd == 0
