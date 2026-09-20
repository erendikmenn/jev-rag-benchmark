from __future__ import annotations

import math
import json
import os
import re
import time
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from .io import read_jsonl, write_jsonl
from .models import Candidate, Document

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class BM25Index:
    """Small deterministic BM25 index used to freeze candidate lists."""

    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_terms = [Counter(tokenize(f"{doc.title} {doc.text}")) for doc in documents]
        self.doc_lengths = [sum(terms.values()) for terms in self.doc_terms]
        self.avg_doc_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.df: Counter[str] = Counter()
        for terms in self.doc_terms:
            self.df.update(terms.keys())

    def retrieve(self, query: str, top_k: int = 20) -> list[Candidate]:
        query_terms = Counter(tokenize(query))
        n_docs = len(self.documents)
        scores: defaultdict[int, float] = defaultdict(float)
        for term, qtf in query_terms.items():
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for idx, terms in enumerate(self.doc_terms):
                tf = terms.get(term, 0)
                if not tf:
                    continue
                norm = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[idx] / max(self.avg_doc_length, 1e-9)
                )
                scores[idx] += qtf * idf * (tf * (self.k1 + 1) / norm)
        ranked = sorted(range(n_docs), key=lambda i: (-scores[i], self.documents[i].doc_id))
        return [
            Candidate(self.documents[idx], float(scores[idx]), rank)
            for rank, idx in enumerate(ranked[:top_k], start=1)
        ]

    def save_documents(self, path: str | Path) -> None:
        write_jsonl(path, self.documents)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Index":
        docs = [Document(**row) for row in read_jsonl(path)]
        return cls(docs)


class EmbeddingProvider(Protocol):
    last_cost_usd: float
    last_input_tokens: int | None
    resolved_model: str | None

    def embed(self, texts: list[str], *, input_type: str | None = None) -> list[list[float]]: ...


class OpenRouterEmbeddings:
    endpoint = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        model: str = "baai/bge-m3",
        *,
        input_usd_per_million_tokens: float = 0.01,
        batch_size: int = 64,
        timeout_seconds: float = 120,
        max_retries: int = 2,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.price = input_usd_per_million_tokens
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = client
        self.last_cost_usd = 0.0
        self.last_input_tokens: int | None = 0
        self.resolved_model: str | None = None

    def embed(self, texts: list[str], *, input_type: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        client = self.client or httpx.Client(timeout=self.timeout_seconds)
        vectors: list[list[float]] = []
        total_tokens = 0
        total_cost = 0.0
        try:
            for start in range(0, len(texts), self.batch_size):
                body: dict[str, Any] = {
                    "model": self.model,
                    "input": texts[start : start + self.batch_size],
                    "encoding_format": "float",
                }
                if input_type:
                    body["input_type"] = input_type
                response = None
                for attempt in range(self.max_retries + 1):
                    try:
                        response = client.post(
                            self.endpoint,
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "HTTP-Referer": "https://github.com/erendikmenn/jev-rag-benchmark",
                                "X-OpenRouter-Title": "jev-rag-benchmark",
                            },
                            json=body,
                        )
                    except httpx.TransportError:
                        if attempt >= self.max_retries:
                            raise
                        time.sleep(min(2**attempt, 10.0))
                        continue
                    if response.status_code < 400:
                        break
                    retryable = response.status_code == 429 or response.status_code >= 500
                    if not retryable or attempt >= self.max_retries:
                        response.raise_for_status()
                    time.sleep(min(float(response.headers.get("retry-after", 2**attempt)), 10.0))
                assert response is not None
                response.raise_for_status()
                payload = response.json()
                ordered = sorted(payload["data"], key=lambda item: item["index"])
                vectors.extend(item["embedding"] for item in ordered)
                usage = payload.get("usage") or {}
                tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
                total_tokens += tokens
                total_cost += float(usage.get("cost") or (tokens * self.price / 1_000_000))
                self.resolved_model = payload.get("model", self.model)
        finally:
            if self.client is None:
                client.close()
        self.last_input_tokens = total_tokens
        self.last_cost_usd = total_cost
        return vectors


def _document_fingerprint(documents: list[Document], model: str) -> str:
    digest = sha256(model.encode("utf-8"))
    for document in documents:
        digest.update(document.doc_id.encode("utf-8"))
        digest.update(document.title.encode("utf-8"))
        digest.update(document.text.encode("utf-8"))
    return digest.hexdigest()


class DenseIndex:
    def __init__(
        self,
        documents: list[Document],
        embedder: EmbeddingProvider,
        *,
        cache_path: str | Path | None = None,
    ):
        self.documents = documents
        self.embedder = embedder
        self.cache_path = Path(cache_path) if cache_path else None
        self.index_cost_usd = 0.0
        self.index_input_tokens = 0
        self.last_query_cost_usd = 0.0
        self.last_query_input_tokens = 0
        self.matrix = self._load_or_build()

    def _load_or_build(self) -> np.ndarray:
        fingerprint = _document_fingerprint(self.documents, self.embedder.model)
        if self.cache_path and self.cache_path.exists():
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == fingerprint:
                return self._normalize(np.asarray(payload["vectors"], dtype=np.float32))

        texts = [f"{document.title}\n{document.text}".strip() for document in self.documents]
        vectors = self.embedder.embed(texts, input_type="search_document")
        self.index_cost_usd = self.embedder.last_cost_usd
        self.index_input_tokens = int(self.embedder.last_input_tokens or 0)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fingerprint": fingerprint,
                "model": self.embedder.model,
                "document_ids": [document.doc_id for document in self.documents],
                "vectors": vectors,
            }
            temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            temporary.replace(self.cache_path)
        return self._normalize(np.asarray(vectors, dtype=np.float32))

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def retrieve(self, query: str, top_k: int = 20) -> list[Candidate]:
        vector = np.asarray(
            self.embedder.embed([query], input_type="search_query")[0], dtype=np.float32
        )
        self.last_query_cost_usd = self.embedder.last_cost_usd
        self.last_query_input_tokens = int(self.embedder.last_input_tokens or 0)
        return self._rank(vector, top_k)

    def _rank(self, vector: np.ndarray, top_k: int) -> list[Candidate]:
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.matrix @ vector
        ranked = sorted(
            range(len(self.documents)),
            key=lambda idx: (-float(scores[idx]), self.documents[idx].doc_id),
        )
        return [
            Candidate(self.documents[idx], float(scores[idx]), rank)
            for rank, idx in enumerate(ranked[:top_k], start=1)
        ]

    def retrieve_many(self, queries: list[str], top_k: int = 20) -> list[list[Candidate]]:
        vectors = self.embedder.embed(queries, input_type="search_query")
        self.last_query_cost_usd = self.embedder.last_cost_usd
        self.last_query_input_tokens = int(self.embedder.last_input_tokens or 0)
        return [self._rank(np.asarray(vector, dtype=np.float32), top_k) for vector in vectors]


class HybridIndex:
    """Fuse lexical and dense rankings with weighted reciprocal-rank fusion."""

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        *,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        source_k: int = 100,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.documents = bm25.documents
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.source_k = source_k
        self.index_cost_usd = dense.index_cost_usd
        self.index_input_tokens = dense.index_input_tokens
        self.last_query_cost_usd = 0.0
        self.last_query_input_tokens = 0

    def retrieve(self, query: str, top_k: int = 20) -> list[Candidate]:
        source_k = min(len(self.documents), max(top_k, self.source_k))
        lexical = self.bm25.retrieve(query, source_k)
        semantic = self.dense.retrieve(query, source_k)
        self.last_query_cost_usd = self.dense.last_query_cost_usd
        self.last_query_input_tokens = self.dense.last_query_input_tokens
        return self._fuse(lexical, semantic, top_k)

    def _fuse(
        self, lexical: list[Candidate], semantic: list[Candidate], top_k: int
    ) -> list[Candidate]:
        by_id = {document.doc_id: document for document in self.documents}
        scores: defaultdict[str, float] = defaultdict(float)
        for item in lexical:
            scores[item.document.doc_id] += self.bm25_weight / (self.rrf_k + item.retrieval_rank)
        for item in semantic:
            scores[item.document.doc_id] += self.dense_weight / (self.rrf_k + item.retrieval_rank)
        ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return [
            Candidate(by_id[doc_id], scores[doc_id], rank)
            for rank, doc_id in enumerate(ranked[:top_k], start=1)
        ]

    def retrieve_many(self, queries: list[str], top_k: int = 20) -> list[list[Candidate]]:
        source_k = min(len(self.documents), max(top_k, self.source_k))
        lexical = [self.bm25.retrieve(query, source_k) for query in queries]
        semantic = self.dense.retrieve_many(queries, source_k)
        self.last_query_cost_usd = self.dense.last_query_cost_usd
        self.last_query_input_tokens = self.dense.last_query_input_tokens
        return [
            self._fuse(lexical_items, semantic_items, top_k)
            for lexical_items, semantic_items in zip(lexical, semantic)
        ]


def create_retrieval_index(
    documents: list[Document], config: dict[str, Any], *, cache_path: str | Path | None = None
):
    backend = config.get("backend", "bm25")
    if backend == "bm25":
        return BM25Index(documents)
    embedding = config.get("embedding") or {}
    embedder = OpenRouterEmbeddings(
        model=embedding.get("model", "baai/bge-m3"),
        input_usd_per_million_tokens=embedding.get("input_usd_per_million_tokens", 0.01),
        batch_size=embedding.get("batch_size", 64),
        timeout_seconds=embedding.get("timeout_seconds", 120),
        max_retries=embedding.get("max_retries", 2),
    )
    dense = DenseIndex(documents, embedder, cache_path=cache_path)
    if backend == "openrouter_dense":
        return dense
    if backend == "hybrid":
        hybrid = config.get("hybrid") or {}
        return HybridIndex(
            BM25Index(documents),
            dense,
            rrf_k=hybrid.get("rrf_k", 60),
            bm25_weight=hybrid.get("bm25_weight", 1.0),
            dense_weight=hybrid.get("dense_weight", 1.0),
            source_k=hybrid.get("source_k", 100),
        )
    raise ValueError(f"unsupported retrieval backend: {backend}")
