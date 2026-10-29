"""Validate the published run and regenerate its monochrome figures without API calls.

Run: uv run --with matplotlib==3.10.8 python scripts/publish_results.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from statistics import mean

from jev_rag_benchmark.fairness import fingerprint
from jev_rag_benchmark.generation import build_prompt
from jev_rag_benchmark.metrics import exact_match, token_f1, ndcg_at_k, mrr_at_k, recall_at_k
from jev_rag_benchmark.models import Candidate, Document

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/fair-2026-10-29"


def validate():
    rows = [json.loads(line) for line in (DATA / "results.jsonl").read_text().splitlines()]
    pool = json.loads(gzip.decompress((DATA / "frozen-pool.json.gz").read_bytes()))
    by_id = {r["query_id"]: r for r in pool}
    assert len(rows) == len(by_id) == len({r["query_id"] for r in rows}) == 1044
    assert {r["query_id"] for r in rows} == set(by_id)
    digest = fingerprint(pool)
    for row in rows:
        frozen = by_id[row["query_id"]]
        assert row["pool_sha256"] == digest
        assert row["candidate_ids"] == [d["doc_id"] for d in frozen["candidates"]]
        assert row["query"] == frozen["query"] and row["references"] == frozen["references"]
        docs = {d["doc_id"]: Document(**d) for d in frozen["candidates"]}
        for method in ("Jev", "Cohere"):
            ids = row["rerank"][method]["context_ids"]
            assert len(ids) == len(set(ids)) == 5 and set(ids) <= set(docs)
            assert not row["rerank"][method]["fallback"]
        contexts = [
            Candidate(docs[d], 0, i + 1) for i, d in enumerate(row["rerank"]["Jev"]["context_ids"])
        ]
        prompt = build_prompt(
            row["query"], contexts, "Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.", 6000
        )
        assert fingerprint(prompt) == row["prompt_sha256"]
        for model in ("Gemini", "DeepSeek"):
            g = row["generation"][model]
            failed = bool(g["error"]) or g["finish_reason"] == "length"
            assert g["f1"] == (0 if failed else token_f1(g["answer"], row["references"]))
            assert g["em"] == (0 if failed else exact_match(g["answer"], row["references"]))
    summary = json.loads((DATA / "summary.json").read_text())
    for model in ("Gemini", "DeepSeek"):
        assert (
            abs(
                summary["generation"][model]["f1"]
                - mean(r["generation"][model]["f1"] for r in rows)
            )
            < 1e-12
        )
    qrels = json.loads((DATA / "qrels.json").read_text())
    for method in ("Baseline", "Jev", "Cohere"):
        for field, metric in (
            ("recall_at_5", recall_at_k),
            ("ndcg_at_5", ndcg_at_k),
            ("mrr_at_5", mrr_at_k),
        ):
            value = mean(
                metric(
                    row["candidate_ids"][:5]
                    if method == "Baseline"
                    else row["rerank"][method]["context_ids"],
                    set(qrels[row["query_id"]]),
                    5,
                )
                for row in rows
            )
            assert abs(value - summary["reranking"][method][field]) < 1e-12
        cost = (
            0
            if method == "Baseline"
            else sum(row["rerank"][method]["estimated_cost_usd"] for row in rows)
        )
        assert abs(cost - summary["reranking"][method]["cost_usd"]) < 1e-9
    for model in ("Gemini", "DeepSeek"):
        gs = [row["generation"][model] for row in rows]
        assert abs(mean(g["em"] for g in gs) - summary["generation"][model]["em"]) < 1e-12
        assert abs(sum(g["cost_usd"] for g in gs) - summary["generation"][model]["cost_usd"]) < 1e-9
    return rows, summary


def validate_clef():
    """Recompute metrics from provider probabilities and verify frozen-pool parity."""
    folder = ROOT / "results/clef-flash-2026-10-29"
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    pool = json.loads(gzip.decompress((DATA / "frozen-pool.json.gz").read_bytes()))
    by_id = {r["query_id"]: r for r in pool}
    qrels = json.loads((DATA / "qrels.json").read_text())
    assert len(rows) == len({r["query_id"] for r in rows}) == len(by_id) == 1044
    assert {r["query_id"] for r in rows} == set(by_id)
    assert len({r["protocol"] for r in rows}) == 1
    digest = fingerprint(pool)
    for row in rows:
        frozen = by_id[row["query_id"]]
        assert row["pool_sha256"] == digest
        assert row["query"] == frozen["query"]
        assert row["candidate_ids"] == [d["doc_id"] for d in frozen["candidates"]]
        payload = row["response"]
        assert payload["model"] == "cloudflare/clef-flash"
        assert payload["provider"] == "DeepInfra"
        assert 0 < payload["usage"]["input_tokens"] < 24576
        scores = [payload["answers"][f"candidate_{i}"]["noul"] for i in range(20)]
        assert all(0 <= value <= 1 for value in scores)
        order = sorted(range(20), key=lambda i: (-scores[i], i))
        assert row["context_ids"] == [row["candidate_ids"][i] for i in order[:5]]
    result = {
        name: mean(metric(r["context_ids"], set(qrels[r["query_id"]]), 5) for r in rows)
        for name, metric in (
            ("recall_at_5", recall_at_k),
            ("ndcg_at_5", ndcg_at_k),
            ("mrr_at_5", mrr_at_k),
        )
    }
    result["hits"] = sum(bool(set(r["context_ids"]) & set(qrels[r["query_id"]])) for r in rows)
    result["cost_usd"] = sum(r["response"]["usage"]["cost"] for r in rows)
    return result


def draw(path, title, subtitle, panels, footer):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "jev-fair-2026-10-29",
        }
    )
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(16, max(7.5, 3.1 * len(panels) + 2.3)), squeeze=False
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(title, x=0.06, y=0.965, ha="left", fontsize=27, fontweight="bold")
    fig.text(0.06, 0.88 if len(panels) == 1 else 0.915, subtitle, fontsize=12, color="#555")
    for ax, (heading, labels, values, unit, maximum) in zip(axes[:, 0], panels):
        ax.barh(labels, values, color=["#777", "#111", "#aaa", "#444"][: len(labels)], height=0.5)
        ax.invert_yaxis()
        ax.set_xlim(0, maximum)
        ax.set_title(heading, loc="left", fontweight="bold", pad=17)
        ax.xaxis.grid(True, linestyle=(0, (4, 5)), color="#d0d0d0")
        ax.set_axisbelow(True)
        ax.set_xlabel(unit)
        for i, value in enumerate(values):
            ax.text(value + maximum * 0.012, i, f"{value:,.2f}", va="center", fontsize=12)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(
        top=0.70 if len(panels) == 1 else 0.85,
        bottom=0.23 if len(panels) == 1 else 0.12,
        left=0.23,
        right=0.92,
        hspace=0.75,
    )
    fig.text(
        0.06,
        0.035,
        footer,
        fontsize=11,
        color="#333",
        bbox={"facecolor": "#f4f4f4", "edgecolor": "#111", "pad": 10},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), metadata={"Date": "2026-10-29"})
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)


def main():
    rows, s = validate()
    r = dict(s["reranking"])
    r["Clef Flash"] = validate_clef()
    g = s["generation"]
    labels = ["Hybrid baseline", "Jev 1.13", "Cohere 3.5", "Clef Flash (DeepInfra)"]
    rerank = [
        (
            "RECALL@5 — HIGHER IS BETTER",
            labels,
            [r[k]["recall_at_5"] * 100 for k in r],
            "Percent (zero-based axis)",
            110,
        ),
        (
            "nDCG@5 — HIGHER IS BETTER",
            labels,
            [r[k]["ndcg_at_5"] * 100 for k in r],
            "Percent (zero-based axis)",
            110,
        ),
        (
            "RERANKING COST — LOWER IS BETTER",
            labels,
            [r[k]["cost_usd"] for k in r],
            "USD / 1,044 queries",
            1.25,
        ),
    ]
    gens = [
        (
            "SHORT-ANSWER TOKEN F1 — HIGHER IS BETTER",
            ["Gemini 3.8 Flash", "DeepSeek V4.1 Flash"],
            [g[k]["f1"] * 100 for k in g],
            "Mean F1 (%)",
            100,
        ),
        (
            "EXACT MATCH — HIGHER IS BETTER",
            ["Gemini 3.8 Flash", "DeepSeek V4.1 Flash"],
            [g[k]["em"] * 100 for k in g],
            "Exact match (%)",
            100,
        ),
        (
            "GENERATION-ONLY COST — LOWER IS BETTER",
            ["Gemini 3.8 Flash", "DeepSeek V4.1 Flash"],
            [g[k]["cost_usd"] for k in g],
            "USD / 1,044 queries",
            3.1,
        ),
    ]
    misses = [
        (
            "GOLD PASSAGE MISSING FROM TOP FIVE — LOWER IS BETTER",
            labels,
            [round((1 - r[k]["recall_at_5"]) * 1044) for k in r],
            "Number of questions",
            max(30, max(round((1 - r[k]["recall_at_5"]) * 1044) for k in r) * 1.2),
        )
    ]
    outputs = [
        ("benchmark/reranker-comparison", "Reranker quality and cost", rerank),
        ("benchmark/generator-tradeoffs", "Short-answer quality and cost", gens),
        ("benchmark/retrieval-errors", "Retrieval errors", misses),
        ("twitter/benchmark-bars", "Retrieval errors", misses),
        ("twitter/benchmark-summary", "Reranker quality and cost", rerank),
        ("twitter/benchmark-details", "Short-answer quality and cost", gens),
    ]
    for name, title, panels in outputs:
        draw(
            ROOT / "assets" / name,
            title,
            "XQUAD-TR  •  29 OCT 2026  •  1,044 QUESTIONS  •  FROZEN HYBRID TOP-20",
            panels,
            "Same Jev top-5 context and prompt for both generators. Six DeepSeek failures score zero. F1 is not human factuality."
            if panels == gens
            else "Same candidate texts and order for all methods. Retrieval preprocessing and pilot overhead excluded from method costs.",
        )
    print("Validated 1,044 paired queries; regenerated 6 SVG + 6 PNG charts.")


if __name__ == "__main__":
    main()
