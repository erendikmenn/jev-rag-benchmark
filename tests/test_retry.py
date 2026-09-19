import httpx

from jev_rag_benchmark.models import Candidate, Document
from jev_rag_benchmark.rerankers import JevReranker


def test_jev_retries_transient_error(monkeypatch):
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"retry-after": "0"})
        return httpx.Response(
            200,
            json={
                "model": "typesafe/jev-1.13-20260917",
                "answers": {"candidate_0": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    monkeypatch.setattr("jev_rag_benchmark.rerankers.time.sleep", lambda _: None)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    candidate = Candidate(Document("d", "useful evidence"), 1.0, 1)
    _, telemetry = JevReranker(
        api_key="test", client=client, max_budget_usd=1, max_retries=1
    ).rerank("question", [candidate], 1)
    assert calls == 2
    assert telemetry.retry_count == 1
