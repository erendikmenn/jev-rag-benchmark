"""Validate pinned Jev 1.14 batch/pointwise requests and recompute matched metrics."""

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
DATA = ROOT / "results/jev114-2026-10-29"


def analyze():
    pool = json.loads(
        gzip.decompress((ROOT / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
    )
    by_id = {r["query_id"]: r for r in pool}
    qrels = json.loads((ROOT / "results/fair-2026-10-29/qrels.json").read_text())
    protocol = json.loads((DATA / "protocol.json").read_text())
    assert protocol["pool"] == fingerprint(pool)
    digest = fingerprint(protocol)
    spec = importlib.util.spec_from_file_location("jev114_runner", ROOT / "scripts/run_jev114.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = [
        json.loads(s)
        for s in gzip.decompress((DATA / "calls.jsonl.gz").read_bytes()).decode().splitlines()
    ]
    pilot = [json.loads(s) for s in (DATA / "pilot.jsonl").read_text().splitlines()]
    grouped = defaultdict(dict)
    for record in calls:
        assert record["protocol"] == digest
        qid, index = record["query_id"], record["index"]
        assert record["request_sha256"] == fingerprint(module.body(by_id[qid], index))
        if not record["ok"]:
            continue
        assert index not in grouped[qid]
        response = record["response"]
        assert response["model"] == "typesafe/jev-1.14-20261014"
        assert response["provider"] == "TypeSafe"
        expected = (
            [response["answers"][f"candidate_{i}"]["noul"] for i in range(20)]
            if index is None
            else [response["answers"]["relevant"]["noul"]]
        )
        assert expected == record["scores"]
        assert all(0 <= score <= 1 for score in expected)
        assert 0 < response["usage"]["input_tokens"] < 32000
        grouped[qid][index] = record
    assert set(grouped) == set(by_id) and len(grouped) == 1044
    rows = []
    for frozen in pool:
        qid = frozen["query_id"]
        gs = grouped[qid]
        assert set(gs) == {None, *range(20)}
        ids = [doc["doc_id"] for doc in frozen["candidates"]]
        row = {"query_id": qid, "candidate_ids": ids, "modes": {}}
        for mode in ("batch", "pointwise"):
            scores = (
                gs[None]["scores"] if mode == "batch" else [gs[i]["scores"][0] for i in range(20)]
            )
            order = sorted(range(20), key=lambda i: (-scores[i], i))
            observations = [gs[None]] if mode == "batch" else [gs[i] for i in range(20)]
            row["modes"][mode] = {
                "scores": scores,
                "context_ids": [ids[i] for i in order[:5]],
                "gold_rank": next(
                    (j + 1 for j, i in enumerate(order) if ids[i] in qrels[qid]), None
                ),
                "cost_usd": sum(r["response"]["usage"]["cost"] for r in observations),
            }
        rows.append(row)
    original = {
        r["query_id"]: r
        for r in map(
            json.loads, (ROOT / "results/fair-2026-10-29/results.jsonl").read_text().splitlines()
        )
    }
    flash = {
        r["query_id"]: r
        for r in map(
            json.loads,
            (ROOT / "results/clef-pointwise-2026-10-29/results.jsonl").read_text().splitlines(),
        )
    }
    modes = {}
    for mode in ("batch", "pointwise"):

        def metric(fn):
            return mean(
                fn(r["modes"][mode]["context_ids"], set(qrels[r["query_id"]]), 5) for r in rows
            )

        summary = {
            "questions": len(rows),
            "gold_rank_1_hits": sum(r["modes"][mode]["gold_rank"] == 1 for r in rows),
            "hits": sum(
                bool(set(r["modes"][mode]["context_ids"]) & set(qrels[r["query_id"]])) for r in rows
            ),
            "recall_at_5": metric(recall_at_k),
            "ndcg_at_5": metric(ndcg_at_k),
            "mrr_at_5": metric(mrr_at_k),
            "cost_usd": sum(r["modes"][mode]["cost_usd"] for r in rows),
            "paired_hits": {},
        }
        for comparator in ("Jev 1.13 batch", "Cohere", "Clef Flash pointwise"):
            wins = losses = 0
            for r in rows:
                qid = r["query_id"]
                other = (
                    flash[qid]["context_ids"]
                    if comparator == "Clef Flash pointwise"
                    else original[qid]["rerank"][
                        "Jev" if comparator == "Jev 1.13 batch" else "Cohere"
                    ]["context_ids"]
                )
                a = bool(set(other) & set(qrels[qid]))
                b = bool(set(r["modes"][mode]["context_ids"]) & set(qrels[qid]))
                wins += b and not a
                losses += a and not b
            summary["paired_hits"][comparator] = {
                "jev114_only_hit": wins,
                "comparator_only_hit": losses,
            }
        modes[mode] = summary
    charged = sum(r["charged_or_reserved_usd"] for r in calls + pilot)
    return rows, {
        "model": "typesafe/jev-1.14-20261014",
        "provider": "TypeSafe",
        "batch_vs_pointwise": {
            "pointwise_better_gold_rank": sum(
                r["modes"]["pointwise"]["gold_rank"] < r["modes"]["batch"]["gold_rank"]
                for r in rows
                if r["modes"]["batch"]["gold_rank"] is not None
            ),
            "batch_better_gold_rank": sum(
                r["modes"]["batch"]["gold_rank"] < r["modes"]["pointwise"]["gold_rank"]
                for r in rows
                if r["modes"]["batch"]["gold_rank"] is not None
            ),
        },
        "modes": modes,
        "attempts": len(calls),
        "failed_attempts": sum(not r["ok"] for r in calls),
        "pilot_cost_usd": sum(r["charged_or_reserved_usd"] for r in pilot),
        "charged_or_reserved_usd": charged,
        "project_total_conservative_usd": 5.507848008360016 + charged,
        "project_cap_usd": 7,
        "gold_present_in_top20": sum(
            bool(set(r["candidate_ids"]) & set(qrels[r["query_id"]])) for r in rows
        ),
        "note": "Same reused evaluation set. Batch compares versions at fixed prompt; pointwise compares models at fixed prompt. No new retrieval or generation.",
    }


if __name__ == "__main__":
    print(json.dumps(analyze()[1], indent=2))
