from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any

from .generation import build_prompt, create_generator
from .io import read_jsonl, write_csv, write_jsonl
from .metrics import exact_match, mrr_at_k, ndcg_at_k, recall_at_k, token_f1
from .models import Document, Query
from .rerankers import (
    BudgetLedger,
    CrossEncoderReranker,
    FixtureJevReranker,
    IdentityReranker,
    JevEvidenceRouter,
    JevReranker,
)
from .retrieval import create_retrieval_index
from .verification import JevCitationVerifier


def load_dataset(documents_path: Path, queries_path: Path, split: str | None = None):
    documents = [Document(**row) for row in read_jsonl(documents_path)]
    queries = [
        Query(
            **{
                **row,
                "relevant_doc_ids": tuple(row.get("relevant_doc_ids", ())),
                "answers": tuple(row.get("answers", ())),
            }
        )
        for row in read_jsonl(queries_path)
    ]
    if split:
        queries = [query for query in queries if query.split == split]
    return documents, queries


def _citations(answer: str) -> set[str]:
    return set(re.findall(r"\[([^\]]+)\]", answer))


def _answer_text(answer: str) -> str:
    """Remove machine-readable citations before QA text metrics."""
    return re.sub(r"\s*\[[^\]]+\]", "", answer).strip()


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
    if config["run"].get("deduplicate_queries", False):
        unique_queries = []
        seen_texts = set()
        for query in queries:
            normalized = " ".join(query.text.casefold().split())
            if normalized not in seen_texts:
                unique_queries.append(query)
                seen_texts.add(normalized)
        queries = unique_queries
    if limit and len(queries) > limit:
        queries = rng.sample(queries, limit)
    index_started = time.perf_counter()
    embedding_model = config["retrieval"].get("embedding", {}).get("model", "none")
    cache_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", embedding_model)
    index = create_retrieval_index(
        documents,
        config["retrieval"],
        cache_path=documents_path.parent / f"embeddings.{cache_name}.json",
    )
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
            strategy="batch",
            max_concurrency=jev_cfg.get("max_concurrency", 12),
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
            strategy="batch",
            max_concurrency=jev_cfg.get("max_concurrency", 12),
        ),
        "P": FixtureJevReranker()
        if fixture_jev
        else JevReranker(
            model=jev_cfg["model"],
            threshold=None,
            timeout_seconds=jev_cfg["timeout_seconds"],
            input_usd_per_million_tokens=jev_cfg["input_usd_per_million_tokens"],
            max_budget_usd=config["run"]["max_budget_usd"],
            budget_ledger=budget_ledger,
            max_retries=jev_cfg["max_retries"],
            strategy="pointwise",
            max_concurrency=jev_cfg.get("max_concurrency", 12),
        ),
        "C": FixtureJevReranker()
        if fixture_jev
        else JevReranker(
            model=jev_cfg["model"],
            threshold=jev_cfg["threshold"],
            timeout_seconds=jev_cfg["timeout_seconds"],
            input_usd_per_million_tokens=jev_cfg["input_usd_per_million_tokens"],
            max_budget_usd=config["run"]["max_budget_usd"],
            budget_ledger=budget_ledger,
            max_retries=jev_cfg["max_retries"],
            strategy="choice",
            max_concurrency=jev_cfg.get("max_concurrency", 12),
        ),
        "R": FixtureJevReranker()
        if fixture_jev
        else JevEvidenceRouter(
            model=jev_cfg["model"],
            timeout_seconds=jev_cfg["timeout_seconds"],
            input_usd_per_million_tokens=jev_cfg["input_usd_per_million_tokens"],
            max_budget_usd=config["run"]["max_budget_usd"],
            budget_ledger=budget_ledger,
            max_retries=jev_cfg["max_retries"],
            max_concurrency=jev_cfg.get("max_concurrency", 12),
            **jev_cfg.get("evidence_router", {}),
        ),
        "V": FixtureJevReranker()
        if fixture_jev
        else JevEvidenceRouter(
            model=jev_cfg["model"],
            timeout_seconds=jev_cfg["timeout_seconds"],
            input_usd_per_million_tokens=jev_cfg["input_usd_per_million_tokens"],
            max_budget_usd=config["run"]["max_budget_usd"],
            budget_ledger=budget_ledger,
            max_retries=jev_cfg["max_retries"],
            max_concurrency=jev_cfg.get("max_concurrency", 12),
            **jev_cfg.get("evidence_router", {}),
        ),
    }
    generator = create_generator(config["generator"], budget_ledger)
    verification_cfg = config.get("verification", {})
    verifier = JevCitationVerifier(
        model=jev_cfg["model"],
        auto_accept_confidence=verification_cfg.get("auto_accept_confidence", 0.8),
        timeout_seconds=jev_cfg["timeout_seconds"],
        input_usd_per_million_tokens=jev_cfg["input_usd_per_million_tokens"],
        max_budget_usd=config["run"]["max_budget_usd"],
        budget_ledger=budget_ledger,
        max_retries=jev_cfg["max_retries"],
    )
    rows = []
    for query_idx, query in enumerate(queries):
        retrieval_started = time.perf_counter()
        candidates = index.retrieve(query.text, candidate_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        retrieval_cost = float(getattr(index, "last_query_cost_usd", 0.0))
        retrieval_input_tokens = int(getattr(index, "last_query_input_tokens", 0))
        ordered_branches = (
            branches[query_idx % len(branches) :] + branches[: query_idx % len(branches)]
        )
        for branch in ordered_branches:
            contexts, telemetry = rerankers[branch].rerank(query.text, candidates, context_k)
            answer = ""
            generation_ms = 0.0
            generation_model = None
            generation_error = None
            prompt_tokens = None
            completion_tokens = None
            generation_cost = 0.0
            verification_ms = 0.0
            verification_cost = 0.0
            verification_input_tokens = 0
            verification_output_tokens = 0
            verification_passed = None
            verification_checks = []
            regeneration_count = 0
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
                generation_cost = generated.cost_usd
                if branch in verification_cfg.get("enabled_branches", []):
                    verified = verifier.verify(
                        answer,
                        contexts,
                        config["policies"]["answer_abstention_text"],
                    )
                    verification_ms += verified.latency_ms
                    verification_cost += verified.cost_usd
                    verification_input_tokens += verified.input_tokens
                    verification_output_tokens += verified.output_tokens
                    verification_passed = verified.passed
                    verification_checks = verified.checks
                    if (
                        not verified.passed
                        and not verified.error
                        and verification_cfg.get("regenerate_once", True)
                    ):
                        feedback = json.dumps(verified.checks, ensure_ascii=False)
                        corrective_prompt = (
                            prompt
                            + "\n\nPREVIOUS ANSWER\n"
                            + answer
                            + "\n\nCITATION VERIFICATION FAILURES\n"
                            + feedback
                            + "\nRewrite the answer once. Remove unsupported claims and use only citations that directly support each factual sentence."
                        )
                        regenerated = generator.generate(corrective_prompt)
                        regeneration_count = 1
                        answer = regenerated.answer
                        generation_ms += regenerated.latency_ms
                        generation_model = regenerated.model
                        generation_error = regenerated.error
                        prompt_tokens = (prompt_tokens or 0) + (regenerated.prompt_tokens or 0)
                        completion_tokens = (completion_tokens or 0) + (
                            regenerated.completion_tokens or 0
                        )
                        generation_cost += regenerated.cost_usd
                        reverified = verifier.verify(
                            answer,
                            contexts,
                            config["policies"]["answer_abstention_text"],
                        )
                        verification_ms += reverified.latency_ms
                        verification_cost += reverified.cost_usd
                        verification_input_tokens += reverified.input_tokens
                        verification_output_tokens += reverified.output_tokens
                        verification_passed = reverified.passed
                        verification_checks = reverified.checks
            ranked_ids = [item.document.doc_id for item in contexts]
            candidate_ids = [item.document.doc_id for item in candidates]
            relevant = set(query.relevant_doc_ids)
            cited = _citations(answer)
            answer_text = _answer_text(answer)
            context_ids = set(ranked_ids)
            support = (
                any(
                    reference.casefold() in item.document.text.casefold()
                    for reference in query.answers
                    for item in contexts
                )
                if query.answers
                else None
            )
            row = {
                "query_id": query.query_id,
                "query": query.text,
                "language": query.language,
                "split": query.split,
                "branch": branch,
                "candidate_ids": candidate_ids,
                "context_ids": ranked_ids,
                "context_routes": [item.route for item in contexts],
                "context_signals": [item.signals for item in contexts],
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
                "answer_em": exact_match(answer_text, query.answers) if query.answers else None,
                "answer_f1": token_f1(answer_text, query.answers) if query.answers else None,
                "source_contains_gold": support,
                "citation_validity": (
                    len(cited & context_ids) / len(cited) if cited else (0.0 if answer else None)
                ),
                "abstained": config["policies"]["answer_abstention_text"] in answer,
                "wrong_answer": bool(
                    query.answers
                    and answer_text
                    and config["policies"]["answer_abstention_text"] not in answer
                    and token_f1(answer_text, query.answers) < 0.5
                ),
                "index_ms_once": index_ms if query_idx == 0 and branch == branches[0] else None,
                "retrieval_ms": retrieval_ms,
                "retrieval_input_tokens": retrieval_input_tokens,
                "retrieval_cost_usd": retrieval_cost,
                "embedding_index_input_tokens_once": (
                    int(getattr(index, "index_input_tokens", 0))
                    if query_idx == 0 and branch == branches[0]
                    else None
                ),
                "embedding_index_cost_usd_once": (
                    float(getattr(index, "index_cost_usd", 0.0))
                    if query_idx == 0 and branch == branches[0]
                    else None
                ),
                "rerank_ms": telemetry.latency_ms,
                "generation_ms": generation_ms,
                "verification_ms": verification_ms,
                "end_to_end_ms": (
                    retrieval_ms + telemetry.latency_ms + generation_ms + verification_ms
                ),
                "reranker_requested_model": telemetry.requested_model,
                "reranker_resolved_model": telemetry.resolved_model,
                "generator_model": generation_model,
                "reranker_input_tokens": telemetry.input_tokens,
                "reranker_output_tokens": telemetry.output_tokens,
                "generator_prompt_tokens": prompt_tokens,
                "generator_completion_tokens": completion_tokens,
                "verification_input_tokens": verification_input_tokens,
                "verification_output_tokens": verification_output_tokens,
                "reranker_cost_usd": telemetry.estimated_cost_usd,
                "generator_cost_usd": generation_cost,
                "verification_cost_usd": verification_cost,
                "online_cost_usd": retrieval_cost
                + telemetry.estimated_cost_usd
                + generation_cost
                + verification_cost,
                "citation_verification_passed": verification_passed,
                "citation_support_fraction": (
                    sum(check["verdict"] == "verified" for check in verification_checks)
                    / len(verification_checks)
                    if verification_checks
                    else None
                ),
                "verification_checks": verification_checks,
                "regeneration_count": regeneration_count,
                "fallback": telemetry.fallback,
                "retry_count": telemetry.retry_count,
                "reranker_error": telemetry.error,
                "reranker_details": telemetry.details,
                "generation_error": generation_error,
                "run_kind": "fixture" if fixture_jev else "real",
            }
            rows.append(row)
        completed = query_idx + 1
        if completed % 25 == 0 or completed == len(queries):
            print(f"progress: {completed}/{len(queries)} queries", flush=True)
    write_jsonl(output_path, rows)
    flat_rows = [
        {
            **row,
            "candidate_ids": "|".join(row["candidate_ids"]),
            "context_ids": "|".join(row["context_ids"]),
            "context_routes": "|".join(row["context_routes"]),
            "context_signals": str(row["context_signals"]),
            "reranker_details": str(row["reranker_details"]),
            "verification_checks": str(row["verification_checks"]),
            "references": "|".join(row["references"]),
        }
        for row in rows
    ]
    write_csv(output_path.with_suffix(".csv"), flat_rows)
    return rows
