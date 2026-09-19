from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .cli import DEFAULT_CONFIG, _dataset_paths
from .config import load_config
from .generation import OllamaGenerator, build_prompt
from .io import read_jsonl
from .models import Document
from .rerankers import IdentityReranker, JevReranker
from .retrieval import BM25Index

app = FastAPI(title="Jev RAG Benchmark API", version="0.1.0")


class QueryRequest(BaseModel):
    query: str
    dataset: str = "xquad-tr"
    branch: str = "A"


@lru_cache(maxsize=3)
def load_index(dataset: str) -> BM25Index:
    documents_path, _ = _dataset_paths(dataset)
    if not documents_path.exists():
        raise FileNotFoundError("dataset missing; run `jev-rag data prepare`")
    return BM25Index([Document(**row) for row in read_jsonl(documents_path)])


@app.post("/query")
def query_documents(request: QueryRequest):
    """Query the same controlled pipeline used by the CLI."""
    config = load_config(DEFAULT_CONFIG)
    try:
        index = load_index(request.dataset)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    candidates = index.retrieve(request.query, config["retrieval"]["candidate_k"])
    if request.branch == "A":
        reranker = IdentityReranker()
    elif request.branch in {"D", "E"}:
        jev = config["rerankers"]["jev"]
        reranker = JevReranker(
            model=jev["model"],
            threshold=jev["threshold"] if request.branch == "E" else None,
            input_usd_per_million_tokens=jev["input_usd_per_million_tokens"],
            max_budget_usd=config["run"]["max_budget_usd"],
        )
    else:
        raise HTTPException(status_code=400, detail="API supports branches A, D, and E")
    contexts, telemetry = reranker.rerank(
        request.query, candidates, config["retrieval"]["context_k"]
    )
    generated = OllamaGenerator(
        config["generator"]["model"],
        config["generator"]["timeout_seconds"],
        config["generator"]["max_output_tokens"],
    ).generate(
        build_prompt(request.query, contexts, config["policies"]["answer_abstention_text"])
    )
    if generated.error:
        raise HTTPException(status_code=503, detail=generated.error)
    return {
        "response": generated.answer,
        "sources": [candidate.document.doc_id for candidate in contexts],
        "branch": request.branch,
        "reranker": telemetry.__dict__,
    }
