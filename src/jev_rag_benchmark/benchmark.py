from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import Any

from .generation import OllamaGenerator, build_prompt
from .io import read_jsonl, write_csv, write_jsonl
from .metrics import exact_match, mrr_at_k, ndcg_at_k, recall_at_k, token_f1
from .models import Document, Query
from .rerankers import (
    BudgetLedger,
    CrossEncoderReranker,
    FixtureJevReranker,
    IdentityReranker,
    JevReranker,
)
from .retrieval import BM25Index


def load_dataset(documents_path: Path, queries_path: Path, split: str | None = None):
    documents = [Document(**row) for row in read_jsonl(documents_path)]
    queries = [Query(**{**row, "relevant_doc_ids": tuple(row.get("relevant_doc_ids", ())), "answers": tuple(row.get("answers", ()))}) for row in read_jsonl(queries_path)]
    if split:
        queries = [query for query in queries if query.split == split]
    return documents, queries


def _citations(answer: str) -> set[str]:
    return set(re.findall(r"\[([^\]]+)\]", answer))


def run_benchmark(
    *,
    documents_path: Path,
    queries_path: Path,
    output_path: Path,
    config: dict[str, Any],
    branches: list[str],
    split: str | None,
    limit: int | None,
    fixture_jev: bool = False,
    skip_generation: bool = False,
) -> list[dict[str, Any]]:
    documents, queries = load_dataset(documents_path, queries_path, split)
    rng = random.Random(config["run"]["seed"])
    queries = sorted(queries, key=lambda q: q.query_id)
    if limit and len(queries) > limit:
        queries = rng.sample(queries, limit)
    index_started = time.perf_counter()
    index = BM25Index(documents)
    index_ms = (time.perf_counter() - index_started) * 1000

    candidate_k = config["retrieval"]["candidate_k"]
    context_k = config["retrieval"]["context_k"]
    jev_cfg = config["rerankers"]["jev"]
    budget_ledger = BudgetLedger(config["run"]["max_budget_usd"])
    rerankers = {
        "A": IdentityReranker(),
        "B": CrossEncoderReranker(**config["rerankers"]["cross_encoder"]),
        "D": FixtureJevReranker()
        if fixture_jev
        else JevReranker(
            model=jev_cfg["model"],
            threshold=None,
            timeout_seconds=jev_cfg["timeout_seconds"],
            input_usd_per_million_tokens=jev_cfg["input_usd_per_million_tokens"],
            max_budget_usd=config["run"]["max_budget_usd"],
            budget_ledger=budget_ledger,
            max_retries=jev_cfg["max_retries"],
        ),
        "E": FixtureJevReranker()
        if fixture_jev
        else JevReranker(
            model=jev_cfg["model"],
            threshold=jev_cfg["threshold"],
            timeout_seconds=jev_cfg["timeout_seconds"],
            input_usd_per_million_tokens=jev_cfg["input_usd_per_million_tokens"],
            max_budget_usd=config["run"]["max_budget_usd"],
            budget_ledger=budget_ledger,
            max_retries=jev_cfg["max_retries"],
        ),
    }
    generator = OllamaGenerator(
        config["generator"]["model"],
        config["generator"]["timeout_seconds"],
        config["generator"]["max_output_tokens"],
    )
    rows = []
    for query_idx, query in enumerate(queries):
        retrieval_started = time.perf_counter()
        candidates = index.retrieve(query.text, candidate_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        ordered_branches = branches[query_idx % len(branches) :] + branches[: query_idx % len(branches)]
        for branch in ordered_branches:
            contexts, telemetry = rerankers[branch].rerank(query.text, candidates, context_k)
            answer = ""
            generation_ms = 0.0
            generation_model = None
            generation_error = None
            prompt_tokens = None
            completion_tokens = None
            if not skip_generation:
                prompt = build_prompt(
                    query.text,
                    contexts,
                    config["policies"]["answer_abstention_text"],
                    config["retrieval"]["context_char_budget"],
                )
                generated = generator.generate(prompt)
                answer = generated.answer
                generation_ms = generated.latency_ms
                generation_model = generated.model
                generation_error = generated.error
                prompt_tokens = generated.prompt_tokens
                completion_tokens = generated.completion_tokens
            ranked_ids = [item.document.doc_id for item in contexts]
            candidate_ids = [item.document.doc_id for item in candidates]
            relevant = set(query.relevant_doc_ids)
            cited = _citations(answer)
            context_ids = set(ranked_ids)
            support = any(
                reference.casefold() in item.document.text.casefold()
                for reference in query.answers
                for item in contexts
            ) if query.answers else None
            row = {
                "query_id": query.query_id,
                "query": query.text,
                "language": query.language,
                "split": query.split,
                "branch": branch,
                "candidate_ids": candidate_ids,
                "context_ids": ranked_ids,
                "context_chars": min(
                    sum(len(item.document.text) for item in contexts),
                    config["retrieval"]["context_char_budget"],
                ),
                "answer": answer,
                "references": list(query.answers),
                "ndcg_10_candidates": ndcg_at_k(candidate_ids, relevant, 10),
                "mrr_10_candidates": mrr_at_k(candidate_ids, relevant, 10),
                "recall_candidate_k": recall_at_k(candidate_ids, relevant, candidate_k),
                "ndcg_10_context": ndcg_at_k(ranked_ids, relevant, 10),
                "mrr_10_context": mrr_at_k(ranked_ids, relevant, 10),
                "recall_context_k": recall_at_k(ranked_ids, relevant, context_k),
                "answer_em": exact_match(answer, query.answers) if query.answers else None,
                "answer_f1": token_f1(answer, query.answers) if query.answers else None,
                "source_contains_gold": support,
                "citation_validity": (
                    len(cited & context_ids) / len(cited) if cited else (0.0 if answer else None)
                ),
                "abstained": config["policies"]["answer_abstention_text"] in answer,
                "wrong_answer": bool(query.answers and answer and not exact_match(answer, query.answers)),
                "index_ms_once": index_ms if query_idx == 0 and branch == branches[0] else None,
                "retrieval_ms": retrieval_ms,
                "rerank_ms": telemetry.latency_ms,
                "generation_ms": generation_ms,
                "end_to_end_ms": retrieval_ms + telemetry.latency_ms + generation_ms,
                "reranker_requested_model": telemetry.requested_model,
                "reranker_resolved_model": telemetry.resolved_model,
                "generator_model": generation_model,
                "reranker_input_tokens": telemetry.input_tokens,
                "reranker_output_tokens": telemetry.output_tokens,
                "generator_prompt_tokens": prompt_tokens,
                "generator_completion_tokens": completion_tokens,
                "online_cost_usd": telemetry.estimated_cost_usd,
                "fallback": telemetry.fallback,
                "retry_count": telemetry.retry_count,
                "reranker_error": telemetry.error,
                "generation_error": generation_error,
                "run_kind": "fixture" if fixture_jev else "real",
            }
            rows.append(row)
    write_jsonl(output_path, rows)
    flat_rows = [
        {**row, "candidate_ids": "|".join(row["candidate_ids"]), "context_ids": "|".join(row["context_ids"]), "references": "|".join(row["references"])}
        for row in rows
    ]
    write_csv(output_path.with_suffix(".csv"), flat_rows)
    return rows
