"""Validate pointwise request identity and compute the frozen-set comparison."""

from __future__ import annotations

import gzip
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from jev_rag_benchmark.fairness import fingerprint
from jev_rag_benchmark.metrics import recall_at_k, ndcg_at_k, mrr_at_k

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/clef-pointwise-2026-10-29"


def analyze():
    pool = json.loads(
        gzip.decompress((ROOT / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
    )
    by_id = {r["query_id"]: r for r in pool}
    qrels = json.loads((ROOT / "results/fair-2026-10-29/qrels.json").read_text())
    protocol = json.loads((DATA / "protocol.json").read_text())
    assert protocol["pool"] == fingerprint(pool)
    digest = fingerprint(protocol)
    spec = importlib.util.spec_from_file_location(
        "pointwise_runner", ROOT / "scripts/run_clef_pointwise.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = [
        json.loads(line)
        for line in gzip.decompress((DATA / "calls.jsonl.gz").read_bytes()).decode().splitlines()
    ]
    grouped = defaultdict(dict)
    for r in calls:
        assert r["protocol"] == digest
        assert r["request_sha256"] == fingerprint(module.body(by_id[r["query_id"]], r["index"]))
        if not r["ok"]:
            continue
        assert r["index"] not in grouped[r["query_id"]], "Duplicate successful observation"
        p = r["response"]
        assert p["model"] == "cloudflare/clef-flash" and p["provider"] == "DeepInfra"
        assert r["score"] == p["answers"]["relevant"]["noul"]
        assert 0 <= r["score"] <= 1
        assert 0 < p["usage"]["input_tokens"] < 24576
        grouped[r["query_id"]][r["index"]] = r
    assert set(grouped) == set(by_id)
    rows = []
    for row in pool:
        qid = row["query_id"]
        gs = grouped[qid]
        assert set(gs) == set(range(20))
        order = sorted(gs, key=lambda i: (-gs[i]["score"], i))
        rows.append(
            {
                "query_id": qid,
                "candidate_ids": [d["doc_id"] for d in row["candidates"]],
                "context_ids": [row["candidates"][i]["doc_id"] for i in order[:5]],
                "gold_rank": next(
                    (
                        j + 1
                        for j, i in enumerate(order)
                        if row["candidates"][i]["doc_id"] in qrels[qid]
                    ),
                    None,
                ),
                "scores": [gs[i]["score"] for i in range(20)],
                "cost_usd": sum(gs[i]["response"]["usage"]["cost"] for i in gs),
            }
        )
    inspected = json.loads(
        (ROOT / "results/clef-flash-diagnostics-2026-10-29/selection.json").read_text()
    )
    inspected = set(inspected["failure_ids"] + inspected["success_ids"])

    def metrics(subset):
        return {
            "questions": len(subset),
            "hits": sum(bool(set(r["context_ids"]) & set(qrels[r["query_id"]])) for r in subset),
            **{
                name: mean(fn(r["context_ids"], set(qrels[r["query_id"]]), 5) for r in subset)
                for name, fn in (
                    ("recall_at_5", recall_at_k),
                    ("ndcg_at_5", ndcg_at_k),
                    ("mrr_at_5", mrr_at_k),
                )
            },
            "cost_usd": sum(r["cost_usd"] for r in subset),
        }

    old = {
        r["query_id"]: r
        for r in map(
            json.loads, (ROOT / "results/fair-2026-10-29/results.jsonl").read_text().splitlines()
        )
    }
    batch = {
        r["query_id"]: r
        for r in map(
            json.loads,
            (ROOT / "results/clef-flash-2026-10-29/results.jsonl").read_text().splitlines(),
        )
    }
    paired = {}
    for method in ("Baseline", "Jev", "Cohere", "Clef batch"):
        wins = losses = 0
        for r in rows:
            qid = r["query_id"]
            comparator = (
                old[qid]["candidate_ids"][:5]
                if method == "Baseline"
                else batch[qid]["context_ids"]
                if method == "Clef batch"
                else old[qid]["rerank"][method]["context_ids"]
            )
            a = bool(set(comparator) & set(qrels[qid]))
            b = bool(set(r["context_ids"]) & set(qrels[qid]))
            wins += b and not a
            losses += a and not b
        paired[method] = {"pointwise_only_hit": wins, "comparator_only_hit": losses}
    summary = metrics(rows)
    summary["gold_present_in_top20"] = sum(
        bool(set(r["candidate_ids"]) & set(qrels[r["query_id"]])) for r in rows
    )
    summary.update(
        {
            "previously_probed_12": metrics([r for r in rows if r["query_id"] in inspected]),
            "other_1032": metrics([r for r in rows if r["query_id"] not in inspected]),
            "paired_hits": paired,
            "attempts": len(calls),
            "failed_attempts": sum(not r["ok"] for r in calls),
            "charged_or_reserved_usd": sum(r["charged_or_reserved_usd"] for r in calls),
            "evaluation_note": "Same evaluation dataset revisited after exploratory diagnostics, not an independent test set. All 1044 rerun, no reused probe scores.",
        }
    )
    return rows, summary


if __name__ == "__main__":
    print(json.dumps(analyze()[1], indent=2))
