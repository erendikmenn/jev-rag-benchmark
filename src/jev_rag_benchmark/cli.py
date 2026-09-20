from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Annotated

import typer

from .benchmark import run_benchmark
from .calibration import CalibrationPoint, query_gate_metrics, select_threshold
from .config import load_config
from .data import prepare_all
from .generation import build_prompt, create_generator
from .io import read_jsonl
from .manifest import build_manifest, write_manifest
from .models import Document
from .metrics import ndcg_at_k, recall_at_k
from .report import generate_report
from .retrieval import BM25Index, create_retrieval_index
from .rerankers import BudgetLedger, IdentityReranker, JevReranker
from .stability import spearman, top_k_jaccard

app = typer.Typer(help="Reproducible Jev RAG benchmark")
data_app = typer.Typer(help="Download and normalize benchmark datasets")
benchmark_app = typer.Typer(help="Run smoke or full benchmarks")
app.add_typer(data_app, name="data")
app.add_typer(benchmark_app, name="benchmark")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/default.yaml"
SMOKE_CONFIG = ROOT / "configs/smoke.yaml"


def _dataset_paths(dataset: str) -> tuple[Path, Path]:
    mapping = {
        "scifact": (
            ROOT / "data/processed/scifact/documents.jsonl",
            ROOT / "data/processed/scifact/queries.test.jsonl",
        ),
        "xquad-en": (
            ROOT / "data/processed/xquad-en/documents.jsonl",
            ROOT / "data/processed/xquad-en/queries.jsonl",
        ),
        "xquad-tr": (
            ROOT / "data/processed/xquad-tr/documents.jsonl",
            ROOT / "data/processed/xquad-tr/queries.jsonl",
        ),
    }
    if dataset not in mapping:
        raise typer.BadParameter(f"unknown dataset: {dataset}")
    return mapping[dataset]


@app.command()
def doctor() -> None:
    """Check local services and credentials without printing secret values."""
    checks = {
        "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
    }
    typer.echo(json.dumps(checks, indent=2))


@data_app.command("prepare")
def data_prepare(force: bool = False) -> None:
    paths = prepare_all(ROOT, force=force)
    for path in paths:
        typer.echo(path.relative_to(ROOT))


@app.command()
def index(dataset: str = "scifact") -> None:
    """Validate and materialize the deterministic BM25 document cache."""
    documents_path, _ = _dataset_paths(dataset)
    documents = [Document(**row) for row in read_jsonl(documents_path)]
    built = BM25Index(documents)
    target = ROOT / f"data/index/{dataset}/documents.jsonl"
    built.save_documents(target)
    typer.echo(f"indexed {len(documents)} documents -> {target.relative_to(ROOT)}")


@app.command()
def estimate(
    dataset: str = "scifact",
    limit: int | None = None,
    jev_branches: int = 1,
    config_path: Path = DEFAULT_CONFIG,
) -> None:
    """Estimate Jev input cost from the actual frozen candidate texts; makes no API calls."""
    config = load_config(config_path)
    documents_path, queries_path = _dataset_paths(dataset)
    documents = [Document(**row) for row in read_jsonl(documents_path)]
    queries = read_jsonl(queries_path)
    if dataset.startswith("xquad"):
        queries = [row for row in queries if row.get("split") == "test"]
    queries = sorted(queries, key=lambda row: row["query_id"])
    if config["run"].get("deduplicate_queries", False):
        unique_queries = []
        seen_texts = set()
        for query in queries:
            normalized = " ".join(query["text"].casefold().split())
            if normalized not in seen_texts:
                unique_queries.append(query)
                seen_texts.add(normalized)
        queries = unique_queries
    if limit is not None and len(queries) > limit:
        queries = random.Random(config["run"]["seed"]).sample(queries, limit)
    index = BM25Index(documents)
    jev_cfg = config["rerankers"]["jev"]
    estimator = JevReranker(
        model=jev_cfg["model"],
        input_usd_per_million_tokens=jev_cfg["input_usd_per_million_tokens"],
    )
    total_tokens = 0
    candidate_k = config["retrieval"]["candidate_k"]
    for query in queries:
        candidates = index.retrieve(query["text"], candidate_k)
        total_tokens += estimator.estimate_input_tokens(query["text"], candidates)
    total_tokens *= max(1, jev_branches)
    total_cost = total_tokens / 1_000_000 * jev_cfg["input_usd_per_million_tokens"]
    typer.echo(
        json.dumps(
            {
                "dataset": dataset,
                "queries": len(queries),
                "candidate_k": candidate_k,
                "jev_branches": jev_branches,
                "estimated_input_tokens": total_tokens,
                "price_usd_per_million_input_tokens": jev_cfg["input_usd_per_million_tokens"],
                "estimated_cost_usd": round(total_cost, 6),
                "calls_made": 0,
            },
            indent=2,
        )
    )


@app.command("retrieval-audit")
def retrieval_audit(
    dataset: str = "xquad-tr",
    candidate_ks: str = "5,10,20,50,100,240",
    split: str | None = "test",
    config_path: Path = DEFAULT_CONFIG,
) -> None:
    """Measure the configured first-stage retrieval ceiling."""
    config = load_config(config_path)
    documents_path, queries_path = _dataset_paths(dataset)
    documents = [Document(**row) for row in read_jsonl(documents_path)]
    queries = read_jsonl(queries_path)
    if split:
        queries = [row for row in queries if row.get("split") == split]

    unique_queries = []
    seen_texts = set()
    for query in sorted(queries, key=lambda row: row["query_id"]):
        normalized = " ".join(query["text"].casefold().split())
        if normalized not in seen_texts:
            unique_queries.append(query)
            seen_texts.add(normalized)

    requested = sorted({int(value.strip()) for value in candidate_ks.split(",") if value.strip()})
    if not requested or requested[0] <= 0:
        raise typer.BadParameter("candidate-ks must contain positive integers")

    embedding_model = config["retrieval"].get("embedding", {}).get("model", "none")
    cache_name = "".join(
        char if char.isalnum() or char in "_.-" else "-" for char in embedding_model
    )
    index = create_retrieval_index(
        documents,
        config["retrieval"],
        cache_path=documents_path.parent / f"embeddings.{cache_name}.json",
    )
    max_k = min(max(requested), len(documents))
    if hasattr(index, "retrieve_many"):
        batches = index.retrieve_many([row["text"] for row in unique_queries], max_k)
        rankings = {
            row["query_id"]: [item.document.doc_id for item in items]
            for row, items in zip(unique_queries, batches)
        }
    else:
        rankings = {
            row["query_id"]: [item.document.doc_id for item in index.retrieve(row["text"], max_k)]
            for row in unique_queries
        }
    results = []
    for k in requested:
        effective_k = min(k, len(documents))
        recalls = []
        ndcgs = []
        hits = 0
        for row in unique_queries:
            relevant = set(row.get("relevant_doc_ids", ()))
            ranked = rankings[row["query_id"]]
            recall = recall_at_k(ranked, relevant, effective_k)
            recalls.append(recall)
            ndcgs.append(ndcg_at_k(ranked, relevant, min(10, effective_k)))
            hits += int(recall > 0)
        results.append(
            {
                "k": effective_k,
                "queries": len(unique_queries),
                "queries_with_relevant": hits,
                "candidate_recall": sum(recalls) / max(1, len(recalls)),
                "ndcg_at_10": sum(ndcgs) / max(1, len(ndcgs)),
            }
        )
    typer.echo(
        json.dumps(
            {
                "dataset": dataset,
                "backend": config["retrieval"].get("backend", "bm25"),
                "embedding_model": (
                    embedding_model if config["retrieval"].get("backend") != "bm25" else None
                ),
                "embedding_index_cost_usd": getattr(index, "index_cost_usd", 0.0),
                "results": results,
            },
            indent=2,
        )
    )


@app.command("calibrate-jev")
def calibrate_jev(
    dataset: str = "xquad-tr",
    config_path: Path = ROOT / "configs/openrouter-hybrid.yaml",
    minimum_recall: float = 0.95,
    limit: int | None = None,
    output: Path = ROOT / "reports/calibration/jev-threshold.json",
) -> None:
    """Select a Jev passage threshold on the development split only."""
    if not 0 < minimum_recall <= 1:
        raise typer.BadParameter("minimum-recall must be in (0, 1]")
    config = load_config(config_path)
    documents_path, queries_path = _dataset_paths(dataset)
    documents = [Document(**row) for row in read_jsonl(documents_path)]
    queries = [row for row in read_jsonl(queries_path) if row.get("split") == "dev"]
    seen_texts = set()
    unique_queries = []
    for query in sorted(queries, key=lambda row: row["query_id"]):
        normalized = " ".join(query["text"].casefold().split())
        if normalized not in seen_texts:
            unique_queries.append(query)
            seen_texts.add(normalized)
    if limit is not None:
        unique_queries = unique_queries[:limit]

    embedding_model = config["retrieval"].get("embedding", {}).get("model", "none")
    cache_name = "".join(
        char if char.isalnum() or char in "_.-" else "-" for char in embedding_model
    )
    index = create_retrieval_index(
        documents,
        config["retrieval"],
        cache_path=documents_path.parent / f"embeddings.{cache_name}.json",
    )
    jev_cfg = config["rerankers"]["jev"]
    ledger = BudgetLedger(config["run"]["max_budget_usd"])
    reranker = JevReranker(
        model=jev_cfg["model"],
        timeout_seconds=jev_cfg["timeout_seconds"],
        input_usd_per_million_tokens=jev_cfg["input_usd_per_million_tokens"],
        max_budget_usd=config["run"]["max_budget_usd"],
        budget_ledger=ledger,
        max_retries=jev_cfg["max_retries"],
        strategy="batch",
    )
    candidate_k = config["retrieval"]["candidate_k"]
    points: list[CalibrationPoint] = []
    traces = []
    for index_number, query in enumerate(unique_queries, start=1):
        candidates = index.retrieve(query["text"], candidate_k)
        scored, telemetry = reranker.rerank(query["text"], candidates, candidate_k)
        if telemetry.fallback:
            raise RuntimeError(f"Jev calibration failed for {query['query_id']}: {telemetry.error}")
        relevant = set(query.get("relevant_doc_ids", ()))
        query_points = [
            CalibrationPoint(
                query_id=query["query_id"],
                doc_id=item.document.doc_id,
                score=float(item.rerank_score or 0.0),
                relevant=item.document.doc_id in relevant,
            )
            for item in scored
        ]
        points.extend(query_points)
        traces.extend(point.__dict__ for point in query_points)
        if index_number % 25 == 0 or index_number == len(unique_queries):
            typer.echo(f"progress: {index_number}/{len(unique_queries)} dev queries", err=True)

    selected = select_threshold(points, minimum_recall=minimum_recall)
    result = {
        "dataset": dataset,
        "split": "dev",
        "queries": len(unique_queries),
        "candidate_k": candidate_k,
        "retrieval_backend": config["retrieval"].get("backend", "bm25"),
        "jev_model": jev_cfg["model"],
        "minimum_recall": minimum_recall,
        "selected": selected,
        "query_gate": query_gate_metrics(points, float(selected["threshold"])),
        "cost_usd": ledger.spent_usd,
        "traces": traces,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.echo(json.dumps({**result, "traces": f"{len(traces)} rows"}, indent=2))


@app.command("jev-stability")
def jev_stability(
    dataset: str = "xquad-tr",
    config_path: Path = ROOT / "configs/openrouter-hybrid.yaml",
    limit: int = 50,
    permutations: int = 5,
    top_k: int = 5,
    output: Path = ROOT / "reports/stability/jev-order-stability.json",
) -> None:
    """Measure Jev batch-score sensitivity to candidate serialization order."""
    if limit <= 0 or permutations < 2 or top_k <= 0:
        raise typer.BadParameter("limit/top-k must be positive and permutations must be >= 2")
    config = load_config(config_path)
    documents_path, queries_path = _dataset_paths(dataset)
    documents = [Document(**row) for row in read_jsonl(documents_path)]
    queries = [row for row in read_jsonl(queries_path) if row.get("split") == "test"]
    unique = []
    seen = set()
    for query in sorted(queries, key=lambda row: row["query_id"]):
        normalized = " ".join(query["text"].casefold().split())
        if normalized not in seen:
            unique.append(query)
            seen.add(normalized)
    unique = unique[:limit]

    embedding_model = config["retrieval"].get("embedding", {}).get("model", "none")
    cache_name = "".join(
        char if char.isalnum() or char in "_.-" else "-" for char in embedding_model
    )
    index = create_retrieval_index(
        documents,
        config["retrieval"],
        cache_path=documents_path.parent / f"embeddings.{cache_name}.json",
    )
    jev_cfg = config["rerankers"]["jev"]
    ledger = BudgetLedger(config["run"]["max_budget_usd"])
    reranker = JevReranker(
        model=jev_cfg["model"],
        timeout_seconds=jev_cfg["timeout_seconds"],
        input_usd_per_million_tokens=jev_cfg["input_usd_per_million_tokens"],
        max_budget_usd=config["run"]["max_budget_usd"],
        budget_ledger=ledger,
        max_retries=jev_cfg["max_retries"],
        strategy="batch",
    )
    rng = random.Random(config["run"]["seed"])
    traces = []
    all_correlations = []
    all_jaccards = []
    gold_flip_count = 0
    for query_number, query in enumerate(unique, start=1):
        candidates = index.retrieve(query["text"], config["retrieval"]["candidate_k"])
        score_runs = []
        gold_hits = []
        relevant = set(query.get("relevant_doc_ids", ()))
        for permutation in range(permutations):
            ordered = list(candidates)
            if permutation:
                rng.shuffle(ordered)
            _, telemetry = reranker.rerank(query["text"], ordered, len(ordered))
            if telemetry.fallback:
                raise RuntimeError(f"Jev stability call failed: {telemetry.error}")
            scores = {key: float(value) for key, value in telemetry.details["scores"].items()}
            score_runs.append(scores)
            top_ids = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
            gold_hits.append(bool(set(top_ids) & relevant))
        correlations = [spearman(score_runs[0], run) for run in score_runs[1:]]
        jaccards = [top_k_jaccard(score_runs[0], run, top_k) for run in score_runs[1:]]
        all_correlations.extend(correlations)
        all_jaccards.extend(jaccards)
        gold_flip_count += int(len(set(gold_hits)) > 1)
        traces.append(
            {
                "query_id": query["query_id"],
                "spearman": correlations,
                "top_k_jaccard": jaccards,
                "gold_in_top_k": gold_hits,
            }
        )
        if query_number % 10 == 0 or query_number == len(unique):
            typer.echo(f"progress: {query_number}/{len(unique)} stability queries", err=True)

    result = {
        "dataset": dataset,
        "queries": len(unique),
        "permutations": permutations,
        "candidate_k": config["retrieval"]["candidate_k"],
        "top_k": top_k,
        "mean_spearman": sum(all_correlations) / max(1, len(all_correlations)),
        "mean_top_k_jaccard": sum(all_jaccards) / max(1, len(all_jaccards)),
        "gold_membership_flip_queries": gold_flip_count,
        "cost_usd": ledger.spent_usd,
        "traces": traces,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.echo(json.dumps({**result, "traces": f"{len(traces)} rows"}, indent=2))


@app.command()
def ask(
    question: Annotated[str, typer.Argument()],
    dataset: str = "xquad-tr",
    branch: str = "A",
    config_path: Path = DEFAULT_CONFIG,
) -> None:
    config = load_config(config_path)
    documents_path, _ = _dataset_paths(dataset)
    index = BM25Index([Document(**row) for row in read_jsonl(documents_path)])
    candidates = index.retrieve(question, config["retrieval"]["candidate_k"])
    ledger = BudgetLedger(config["run"]["max_budget_usd"])
    if branch == "D":
        jev = config["rerankers"]["jev"]
        reranker = JevReranker(
            model=jev["model"],
            input_usd_per_million_tokens=jev["input_usd_per_million_tokens"],
            max_budget_usd=config["run"]["max_budget_usd"],
            budget_ledger=ledger,
        )
    else:
        reranker = IdentityReranker()
    contexts, telemetry = reranker.rerank(question, candidates, config["retrieval"]["context_k"])
    prompt = build_prompt(
        question,
        contexts,
        config["policies"]["answer_abstention_text"],
        config["retrieval"]["context_char_budget"],
    )
    result = create_generator(config["generator"], ledger).generate(prompt)
    typer.echo(result.answer or result.error)
    typer.echo(
        json.dumps(
            {"sources": [c.document.doc_id for c in contexts], "reranker": telemetry.__dict__},
            ensure_ascii=False,
        )
    )


def _run(
    dataset: str,
    config_path: Path,
    branches: str,
    limit: int | None,
    fixture_jev: bool,
    skip_generation: bool,
):
    config = load_config(config_path)
    documents_path, queries_path = _dataset_paths(dataset)
    missing = [path for path in (documents_path, queries_path) if not path.exists()]
    if missing:
        raise typer.BadParameter("dataset missing; run `jev-rag data prepare` first")
    selected = [item.strip().upper() for item in branches.split(",") if item.strip()]
    if "E" in selected and config["rerankers"]["jev"].get("threshold_source") != "dev":
        raise typer.BadParameter(
            "branch E is locked until rerankers.jev.threshold_source is set to dev after calibration"
        )
    name = f"{dataset}-{'-'.join(selected).lower()}"
    output = ROOT / f"results/{name}.jsonl"
    split = "test" if dataset.startswith("xquad") else None
    rows = run_benchmark(
        documents_path=documents_path,
        queries_path=queries_path,
        output_path=output,
        config=config,
        branches=selected,
        split=split,
        limit=limit,
        fixture_jev=fixture_jev,
        skip_generation=skip_generation,
    )
    manifest = build_manifest(
        ROOT, config, [documents_path, queries_path], run_kind="fixture" if fixture_jev else "real"
    )
    manifest["models"]["generator_resolved"] = sorted(
        {row["generator_model"] for row in rows if row.get("generator_model")}
    )
    manifest["models"]["reranker_resolved"] = sorted(
        {row["reranker_resolved_model"] for row in rows if row.get("reranker_resolved_model")}
    )
    write_manifest(output.with_suffix(".manifest.json"), manifest)
    typer.echo(f"wrote {len(rows)} rows -> {output.relative_to(ROOT)}")


@benchmark_app.command("smoke")
def benchmark_smoke(
    dataset: str = "scifact",
    branches: str = "A,B,D",
    fixture_jev: bool = True,
    skip_generation: bool = True,
    config_path: Path = SMOKE_CONFIG,
) -> None:
    _run(dataset, config_path, branches, 25, fixture_jev, skip_generation)


@benchmark_app.command("full")
def benchmark_full(
    dataset: str = "scifact",
    branches: str = "A,B,D",
    limit: int | None = None,
    fixture_jev: bool = False,
    skip_generation: bool = False,
    config_path: Path = DEFAULT_CONFIG,
) -> None:
    _run(dataset, config_path, branches, limit, fixture_jev, skip_generation)


@app.command()
def report(
    results: Path,
    output_dir: Path = ROOT / "reports/generated",
    seed: int = 20260919,
) -> None:
    path = generate_report(results, output_dir, seed)
    typer.echo(path)


if __name__ == "__main__":
    app()
