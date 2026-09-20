import httpx

from jev_rag_benchmark.models import Candidate, Document
from jev_rag_benchmark.rerankers import (
    JevEvidenceRouter,
    JevHierarchicalReranker,
    JevPermutationEnsembleReranker,
    JevReranker,
)


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


def test_pointwise_jev_scores_each_candidate_independently():
    def handler(request: httpx.Request):
        payload = __import__("json").loads(request.content)
        is_ankara = "Ankara" in payload["state"]["candidate"]["text"]
        return httpx.Response(
            200,
            json={
                "id": "pointwise",
                "model": "typesafe/jev-1.13-test",
                "answers": {"relevant": {"type": "noul", "noul": 0.9 if is_ankara else 0.1}},
                "usage": {"input_tokens": 20, "output_tokens": 1, "cost": 0.000001},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    reranker = JevReranker(api_key="test", client=client, max_budget_usd=1, strategy="pointwise")
    selected, telemetry = reranker.rerank("capital Türkiye", candidates(), 1)
    assert selected[0].document.doc_id == "d1"
    assert telemetry.method == "jev_pointwise"
    assert telemetry.input_tokens == 40


def test_evidence_router_drops_injection_and_keeps_conflict():
    def handler(request: httpx.Request):
        payload = __import__("json").loads(request.content)
        is_ankara = "Ankara" in payload["state"]["candidate"]["text"]
        values = {
            "relevant": 0.95,
            "evidence": 0.9 if is_ankara else 0.1,
            "contradiction": 0.1 if is_ankara else 0.9,
            "injection": 0.05,
        }
        return httpx.Response(
            200,
            json={
                "id": "router",
                "model": "typesafe/jev-1.13-test",
                "answers": {key: {"type": "noul", "noul": value} for key, value in values.items()},
                "usage": {"input_tokens": 20, "output_tokens": 4, "cost": 0.000001},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    router = JevEvidenceRouter(api_key="test", client=client, max_budget_usd=1)
    selected, telemetry = router.rerank("capital Türkiye", candidates(), 2)
    assert [item.route for item in selected] == ["evidence", "conflict"]
    assert telemetry.details["route_counts"] == {"evidence": 1, "conflict": 1}


def test_hierarchical_jev_scores_shards_then_finalists():
    def handler(request: httpx.Request):
        payload = __import__("json").loads(request.content)
        answers = {}
        for index, candidate in enumerate(payload["state"]["candidates"]):
            score = 0.9 if candidate["id"] in {"d1", "d3"} else 0.1
            if len(payload["state"]["candidates"]) == 2 and candidate["id"] == "d3":
                score = 0.99
            answers[f"candidate_{index}"] = {"type": "noul", "noul": score}
        return httpx.Response(
            200,
            json={
                "id": "hierarchical",
                "model": "typesafe/jev-1.13-test",
                "answers": answers,
                "usage": {"input_tokens": 20, "output_tokens": 2, "cost": 0.000001},
            },
        )

    docs = [
        Candidate(Document(f"d{index}", f"passage {index}"), 1.0 / index, index)
        for index in range(1, 5)
    ]
    reranker = JevHierarchicalReranker(
        api_key="test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_budget_usd=1,
        shard_size=2,
        shard_top_k=1,
    )
    selected, telemetry = reranker.rerank("query", docs, 1)
    assert selected[0].document.doc_id == "d3"
    assert telemetry.details["shards"] == 2
    assert telemetry.details["finalists"] == 2


def test_permutation_ensemble_averages_multiple_batch_scores():
    request_count = 0

    def handler(request: httpx.Request):
        nonlocal request_count
        request_count += 1
        payload = __import__("json").loads(request.content)
        answers = {
            f"candidate_{index}": {
                "type": "noul",
                "noul": 0.9 if candidate["id"] == "d1" else 0.1,
            }
            for index, candidate in enumerate(payload["state"]["candidates"])
        }
        return httpx.Response(
            200,
            json={
                "id": f"ensemble-{request_count}",
                "model": "typesafe/jev-1.13-test",
                "answers": answers,
                "usage": {"input_tokens": 20, "output_tokens": 2, "cost": 0.000001},
            },
        )

    reranker = JevPermutationEnsembleReranker(
        api_key="test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_budget_usd=1,
        permutations=3,
    )
    selected, telemetry = reranker.rerank("capital Türkiye", candidates(), 1)
    assert selected[0].document.doc_id == "d1"
    assert telemetry.method == "jev_permutation_ensemble"
    assert telemetry.input_tokens == 60
    assert telemetry.estimated_cost_usd == 0.000003
    assert telemetry.details["permutations"] == 3
    assert request_count == 3
