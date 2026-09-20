import httpx

from jev_rag_benchmark.models import Candidate, Document
from jev_rag_benchmark.verification import JevCitationVerifier


def test_citation_verifier_checks_semantic_support():
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "id": "verify",
                "model": "typesafe/jev-1.13-test",
                "answers": {
                    "relation": {
                        "type": "choice",
                        "choice": "supports",
                        "probabilities": {
                            "supports": 0.95,
                            "contradicts": 0.01,
                            "says_nothing": 0.04,
                        },
                        "confidence": 0.94,
                    }
                },
                "usage": {"input_tokens": 30, "output_tokens": 1, "cost": 0.000002},
            },
        )

    context = [Candidate(Document("source-1", "Ankara is Türkiye's capital."), 1.0, 1)]
    verifier = JevCitationVerifier(
        api_key="test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_budget_usd=1,
    )
    result = verifier.verify("Ankara Türkiye'nin başkentidir [source-1].", context, "NO")
    assert result.passed is True
    assert result.verified_fraction == 1.0
    assert result.cost_usd == 0.000002


def test_citation_verifier_rejects_unknown_and_missing_citations():
    context = [Candidate(Document("source-1", "Evidence."), 1.0, 1)]
    verifier = JevCitationVerifier(api_key="test", max_budget_usd=1)
    fabricated = verifier.verify("Claim [missing].", context, "NO")
    uncited = verifier.verify("Claim without a citation.", context, "NO")
    assert fabricated.checks[0]["verdict"] == "fabricated"
    assert uncited.checks[0]["verdict"] == "uncited"
