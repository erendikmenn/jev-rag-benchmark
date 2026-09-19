from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from .benchmark import run_benchmark
from .config import load_config
from .data import prepare_all
from .generation import OllamaGenerator, build_prompt
from .io import read_jsonl
from .manifest import build_manifest, write_manifest
from .models import Document
from .report import generate_report
from .retrieval import BM25Index
from .rerankers import IdentityReranker, JevReranker

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
        "TYPESAFE_API_KEY": bool(os.getenv("TYPESAFE_API_KEY")),
        "ollama_binary": subprocess.run(["which", "ollama"], capture_output=True).returncode == 0,
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
    if limit is not None:
        queries = sorted(queries, key=lambda row: row["query_id"])[:limit]
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
                "price_usd_per_million_input_tokens": jev_cfg[
                    "input_usd_per_million_tokens"
                ],
                "estimated_cost_usd": round(total_cost, 6),
                "calls_made": 0,
            },
            indent=2,
        )
    )


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
    if branch == "D":
        jev = config["rerankers"]["jev"]
        reranker = JevReranker(
            model=jev["model"],
            input_usd_per_million_tokens=jev["input_usd_per_million_tokens"],
            max_budget_usd=config["run"]["max_budget_usd"],
        )
    else:
        reranker = IdentityReranker()
    contexts, telemetry = reranker.rerank(question, candidates, config["retrieval"]["context_k"])
    prompt = build_prompt(question, contexts, config["policies"]["answer_abstention_text"])
    result = OllamaGenerator(
        config["generator"]["model"],
        config["generator"]["timeout_seconds"],
        config["generator"]["max_output_tokens"],
    ).generate(prompt)
    typer.echo(result.answer or result.error)
    typer.echo(json.dumps({"sources": [c.document.doc_id for c in contexts], "reranker": telemetry.__dict__}, ensure_ascii=False))


def _run(dataset: str, config_path: Path, branches: str, limit: int | None, fixture_jev: bool, skip_generation: bool):
    config = load_config(config_path)
    documents_path, queries_path = _dataset_paths(dataset)
    missing = [path for path in (documents_path, queries_path) if not path.exists()]
    if missing:
        raise typer.BadParameter("dataset missing; run `jev-rag data prepare` first")
    selected = [item.strip().upper() for item in branches.split(",") if item.strip()]
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
    manifest = build_manifest(ROOT, config, [documents_path, queries_path], run_kind="fixture" if fixture_jev else "real")
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
