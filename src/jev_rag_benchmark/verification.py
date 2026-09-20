from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx

from .models import Candidate
from .rerankers import BudgetLedger, JevReranker


@dataclass
class CitationCheckResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def verified_fraction(self) -> float:
        if not self.checks:
            return 0.0
        return sum(check["verdict"] == "verified" for check in self.checks) / len(self.checks)


class JevCitationVerifier:
    def __init__(
        self,
        *,
        model: str = "typesafe/jev-1.13",
        auto_accept_confidence: float = 0.8,
        timeout_seconds: float = 60,
        input_usd_per_million_tokens: float = 0.042,
        max_budget_usd: float = 0.0,
        budget_ledger: BudgetLedger | None = None,
        max_retries: int = 2,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.auto_accept_confidence = auto_accept_confidence
        self.engine = JevReranker(
            model=model,
            timeout_seconds=timeout_seconds,
            input_usd_per_million_tokens=input_usd_per_million_tokens,
            max_budget_usd=max_budget_usd,
            budget_ledger=budget_ledger,
            max_retries=max_retries,
            api_key=api_key,
            client=client,
        )

    @staticmethod
    def _claims(answer: str) -> list[tuple[str, str]]:
        claims = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()):
            source_ids = re.findall(r"\[([^\]]+)\]", sentence)
            claim = re.sub(r"\s*\[[^\]]+\]", "", sentence).strip()
            for source_id in source_ids:
                claims.append((claim, source_id))
        return claims

    def verify(
        self, answer: str, contexts: list[Candidate], abstention_text: str
    ) -> CitationCheckResult:
        started = time.perf_counter()
        if not answer or abstention_text in answer:
            return CitationCheckResult(
                passed=True, latency_ms=(time.perf_counter() - started) * 1000
            )
        claims = self._claims(answer)
        if not claims:
            return CitationCheckResult(
                passed=False,
                checks=[
                    {
                        "claim": answer,
                        "source_id": None,
                        "verdict": "uncited",
                        "confidence": None,
                        "auto": True,
                    }
                ],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        sources = {item.document.doc_id: item.document.text for item in contexts}
        checks = []
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        client = self.engine.client or httpx.Client(timeout=self.engine.timeout_seconds)
        try:
            for claim, source_id in claims:
                if source_id not in sources:
                    checks.append(
                        {
                            "claim": claim,
                            "source_id": source_id,
                            "verdict": "fabricated",
                            "confidence": None,
                            "auto": True,
                        }
                    )
                    continue
                state = {"claim": claim, "source": sources[source_id]}
                questions = {
                    "relation": {
                        "type": "choice",
                        "instructions": "How does `source` relate to `claim`?",
                        "criteria": {
                            "supports": "The source states the claim or directly implies that it is true.",
                            "contradicts": "The source states the opposite of the claim or implies it is false.",
                            "says_nothing": "The source does not address what the claim asserts, either way.",
                        },
                    }
                }
                payload, _, _ = self.engine._request(client, state, questions)
                answer_payload = payload["answers"]["relation"]
                choice = answer_payload["choice"]
                confidence = float(
                    answer_payload.get("confidence")
                    or max(answer_payload.get("probabilities", {}).values(), default=0.0)
                )
                verdict = {
                    "supports": "verified",
                    "contradicts": "contradicted",
                    "says_nothing": "unsupported",
                }[choice]
                checks.append(
                    {
                        "claim": claim,
                        "source_id": source_id,
                        "verdict": verdict,
                        "confidence": confidence,
                        "auto": confidence >= self.auto_accept_confidence,
                    }
                )
                estimated = self.engine.estimate_cost(
                    claim, [item for item in contexts if item.document.doc_id == source_id]
                )
                used_in, used_out, used_cost = self.engine._usage(
                    payload, estimated, self.engine.price
                )
                input_tokens += used_in
                output_tokens += used_out
                cost += used_cost
            self.engine.budget_ledger.charge(cost)
            passed = bool(checks) and all(check["verdict"] == "verified" for check in checks)
            return CitationCheckResult(
                passed=passed,
                checks=checks,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        except Exception as exc:
            return CitationCheckResult(
                passed=False,
                checks=checks,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if self.engine.client is None:
                client.close()
