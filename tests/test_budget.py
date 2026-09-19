from jev_rag_benchmark.models import Candidate, Document
from jev_rag_benchmark.rerankers import JevReranker


def test_zero_budget_blocks_paid_call():
    candidate = Candidate(Document("d", "evidence"), 1.0, 1)
    _, telemetry = JevReranker(api_key="present", max_budget_usd=0).rerank(
        "question", [candidate], 1
    )
    assert telemetry.fallback
    assert "max_budget_usd" in (telemetry.error or "")

