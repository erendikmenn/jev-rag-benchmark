from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from jev_rag_benchmark.config import load_config
from jev_rag_benchmark.generation import build_prompt, create_generator
from jev_rag_benchmark.io import read_jsonl, sha256_file, write_csv, write_jsonl
from jev_rag_benchmark.metrics import exact_match, token_f1
from jev_rag_benchmark.models import Candidate, Document
from jev_rag_benchmark.rerankers import BudgetLedger


def answer_text(answer: str) -> str:
    return re.sub(r"\s*\[[^\]]+\]", "", answer).strip()


def citations(answer: str) -> set[str]:
    return set(re.findall(r"\[([^\]]+)\]", answer))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a generator over frozen contexts from an existing benchmark."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("documents", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-branch", default="D")
    parser.add_argument("--target-branch", default="G")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-completion-tokens-at-least", type=int)
    parser.add_argument("--retry-finish-reason")
    args = parser.parse_args()

    source_rows = read_jsonl(args.source)
    frozen = [row for row in source_rows if row["branch"] == args.source_branch]
    if args.limit:
        frozen = frozen[: args.limit]
    documents = {row["doc_id"]: Document(**row) for row in read_jsonl(args.documents)}
    config = load_config(args.config)
    ledger = BudgetLedger(config["run"]["max_budget_usd"])
    generator = create_generator(config["generator"], ledger)
    abstention_text = config["policies"]["answer_abstention_text"]

    completed: dict[str, dict] = {}
    if args.output.exists():
        completed = {}
        for row in read_jsonl(args.output):
            if row["branch"] != args.target_branch or row.get("generation_error"):
                continue
            threshold = args.retry_completion_tokens_at_least
            if threshold and int(row.get("generator_completion_tokens") or 0) >= threshold:
                continue
            if args.retry_finish_reason and row.get("generation_finish_reason") == args.retry_finish_reason:
                continue
            completed[row["query_id"]] = row
        ledger.spent_usd = sum(
            float(row.get("replay_incremental_cost_usd", 0.0)) for row in completed.values()
        )

    generated_rows = dict(completed)
    for index, original in enumerate(frozen, start=1):
        if original["query_id"] in completed:
            continue
        contexts = [
            Candidate(documents[doc_id], retrieval_score=0.0, retrieval_rank=rank)
            for rank, doc_id in enumerate(original["context_ids"], start=1)
        ]
        prompt = build_prompt(
            original["query"],
            contexts,
            abstention_text,
            config["retrieval"]["context_char_budget"],
        )
        result = generator.generate(prompt)
        cleaned = answer_text(result.answer)
        cited = citations(result.answer)
        context_ids = set(original["context_ids"])
        row = {
            **original,
            "branch": args.target_branch,
            "answer": result.answer,
            "answer_em": exact_match(cleaned, original["references"]),
            "answer_f1": token_f1(cleaned, original["references"]),
            "citation_validity": (
                len(cited & context_ids) / len(cited)
                if cited
                else (0.0 if result.answer else None)
            ),
            "abstained": abstention_text in result.answer,
            "wrong_answer": bool(
                original["references"]
                and cleaned
                and abstention_text not in result.answer
                and token_f1(cleaned, original["references"]) < 0.5
            ),
            "generation_ms": result.latency_ms,
            "end_to_end_ms": (
                float(original["retrieval_ms"])
                + float(original["rerank_ms"])
                + result.latency_ms
            ),
            "generator_model": result.model,
            "generator_prompt_tokens": result.prompt_tokens,
            "generator_completion_tokens": result.completion_tokens,
            "generator_cost_usd": result.cost_usd,
            "online_cost_usd": float(original["reranker_cost_usd"]) + result.cost_usd,
            "replay_incremental_cost_usd": result.cost_usd,
            "generation_error": result.error,
            "generation_finish_reason": result.finish_reason,
        }
        generated_rows[original["query_id"]] = row

        if index % 10 == 0 or index == len(frozen) or result.error:
            combined = source_rows + [generated_rows[key] for key in sorted(generated_rows)]
            write_jsonl(args.output, combined)
            print(
                f"progress: {len(generated_rows)}/{len(frozen)}; "
                f"incremental cost: ${ledger.spent_usd:.6f}",
                flush=True,
            )

    combined = source_rows + [generated_rows[key] for key in sorted(generated_rows)]
    write_jsonl(args.output, combined)
    flat_rows = [
        {
            **row,
            "candidate_ids": "|".join(row["candidate_ids"]),
            "context_ids": "|".join(row["context_ids"]),
            "references": "|".join(row["references"]),
        }
        for row in combined
    ]
    write_csv(args.output.with_suffix(".csv"), flat_rows)
    source_manifest_path = args.source.with_suffix(".manifest.json")
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    manifest = {
        **source_manifest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_kind": "real-generator-replay",
        "models": {
            **source_manifest.get("models", {}),
            "generator_requested": config["generator"]["model"],
            "generator_resolved": sorted(
                {row["generator_model"] for row in generated_rows.values()}
            ),
        },
        "prices": {
            **source_manifest.get("prices", {}),
            "generator_input_usd_per_million_tokens": config["generator"][
                "input_usd_per_million_tokens"
            ],
            "generator_output_usd_per_million_tokens": config["generator"][
                "output_usd_per_million_tokens"
            ],
            "generator_verified_on": str(config["generator"]["price_verified_on"]),
        },
        "config": config,
        "replay": {
            "source_results": str(args.source),
            "source_results_sha256": sha256_file(args.source),
            "source_branch": args.source_branch,
            "target_branch": args.target_branch,
            "frozen_contexts": True,
            "adaptive_max_output_tokens": config["generator"].get("retry_token_caps"),
            "rows": len(generated_rows),
        },
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
