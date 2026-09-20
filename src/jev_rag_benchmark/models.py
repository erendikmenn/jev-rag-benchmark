from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    relevant_doc_ids: tuple[str, ...] = ()
    answers: tuple[str, ...] = ()
    language: str = "en"
    split: str = "test"


@dataclass(frozen=True)
class Candidate:
    document: Document
    retrieval_score: float
    retrieval_rank: int
    rerank_score: float | None = None
    route: str = "evidence"
    signals: dict[str, float] = field(default_factory=dict)


@dataclass
class RerankTelemetry:
    method: str
    latency_ms: float
    requested_model: str | None = None
    resolved_model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float = 0.0
    fallback: bool = False
    error: str | None = None
    request_id: str | None = None
    retry_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)
