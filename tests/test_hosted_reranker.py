import httpx

from jev_rag_benchmark.models import Candidate, Document
from jev_rag_benchmark.rerankers import OpenRouterReranker


def test_openrouter_hosted_reranker_uses_returned_indexes_and_cost():
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "id": "rerank-test",
                "model": "rerank-v3.5",
                "provider": "Cohere",
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ],
                "usage": {"search_units": 1, "cost": 0.001},
            },
        )

    candidates = [
        Candidate(Document("d1", "noise"), 1.0, 1),
        Candidate(Document("d2", "answer"), 0.5, 2),
    ]
    reranker = OpenRouterReranker(
        api_key="test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_budget_usd=1,
    )
    selected, telemetry = reranker.rerank("query", candidates, 1)
    assert selected[0].document.doc_id == "d2"
    assert telemetry.estimated_cost_usd == 0.001
    assert telemetry.details["provider"] == "Cohere"
