"""Recompute the Luna Decisions and Haiku/Luna extension from saved raw calls."""

from __future__ import annotations

import gzip
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from jev_rag_benchmark.fairness import fingerprint
from jev_rag_benchmark.metrics import recall_at_k, ndcg_at_k, mrr_at_k, token_f1, exact_match

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/luna-haiku-2026-10-29"


def analyze():
    pool = json.loads(
        gzip.decompress((ROOT / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
    )
    by_id = {r["query_id"]: r for r in pool}
    old = {
        r["query_id"]: r
        for r in map(
            json.loads, (ROOT / "results/fair-2026-10-29/results.jsonl").read_text().splitlines()
        )
    }
    qrels = json.loads((ROOT / "results/fair-2026-10-29/qrels.json").read_text())
    protocol = json.loads((DATA / "protocol.json").read_text())
    assert protocol["pool"] == fingerprint(pool)
    digest = fingerprint(protocol)
    spec = importlib.util.spec_from_file_location(
        "luna_haiku_runner", ROOT / "scripts/run_luna_haiku.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = [
        json.loads(line)
        for line in gzip.decompress((DATA / "calls.jsonl.gz").read_bytes()).decode().splitlines()
    ]
    grouped = defaultdict(dict)
    for record in calls:
        assert record["protocol"] == digest
        qid, mode, index = record["query_id"], record["mode"], record["index"]
        request = module.body(by_id[qid], mode, index, old[qid])
        assert fingerprint(request) == record["request_sha256"]
        if not record["ok"]:
            continue
        assert (mode, index) not in grouped[qid]
        payload = record["response"]
        if "cost" in payload.get("usage", {}):
            assert record["charged_or_reserved_usd"] == payload["usage"]["cost"]
        else:
            assert mode in ("Luna", "Haiku") and record["quality_failure"]
            assert record["charged_or_reserved_usd"] == 0.004
        assert payload["model"] in (module.MODELS[mode][0], module.MODELS[mode][2])
        assert payload["provider"] == module.MODELS[mode][1]
        if mode in ("batch", "pointwise"):
            expected = (
                [payload["answers"][f"candidate_{i}"]["noul"] for i in range(20)]
                if mode == "batch"
                else [payload["answers"]["relevant"]["noul"]]
            )
            assert record["scores"] == expected and all(0 <= s <= 1 for s in expected)
        else:
            choice = payload["choices"][0]
            assert record["answer"] == (choice["message"].get("content") or "").strip()
            assert record["finish_reason"] == choice.get("finish_reason")
            assert record["quality_failure"] == (
                not record["answer"] or record["finish_reason"] == "length"
            )
            assert fingerprint(request["messages"][0]["content"]) == old[qid]["prompt_sha256"]
        grouped[qid][(mode, index)] = record
    assert set(grouped) == set(by_id) and len(grouped) == 1044
    rows = []
    expected_keys = {
        ("batch", None),
        ("Luna", None),
        ("Haiku", None),
        *[("pointwise", i) for i in range(20)],
    }
    for frozen in pool:
        qid = frozen["query_id"]
        gs = grouped[qid]
        assert set(gs) == expected_keys
        ids = [d["doc_id"] for d in frozen["candidates"]]
        row = {"query_id": qid, "candidate_ids": ids, "reranking": {}, "generation": {}}
        for mode in ("batch", "pointwise"):
            observations = (
                [gs[("batch", None)]]
                if mode == "batch"
                else [gs[("pointwise", i)] for i in range(20)]
            )
            scores = (
                observations[0]["scores"]
                if mode == "batch"
                else [r["scores"][0] for r in observations]
            )
            order = sorted(range(20), key=lambda i: (-scores[i], i))
            row["reranking"][mode] = {
                "scores": scores,
                "context_ids": [ids[i] for i in order[:5]],
                "gold_rank": next(
                    (j + 1 for j, i in enumerate(order) if ids[i] in qrels[qid]), None
                ),
                "cost_usd": sum(r["response"]["usage"]["cost"] for r in observations),
            }
        for mode in ("Luna", "Haiku"):
            observation = gs[(mode, None)]
            failed = observation["quality_failure"]
            row["generation"][mode] = {
                "answer": observation["answer"],
                "f1": 0 if failed else token_f1(observation["answer"], frozen["references"]),
                "em": 0 if failed else exact_match(observation["answer"], frozen["references"]),
                "quality_failure": failed,
                "finish_reason": observation["finish_reason"],
                "cost_usd": observation["charged_or_reserved_usd"],
                "cost_is_reserve": "cost" not in observation["response"].get("usage", {}),
                "prompt_tokens": observation["response"].get("usage", {}).get("prompt_tokens"),
                "completion_tokens": observation["response"]
                .get("usage", {})
                .get("completion_tokens"),
                "prompt_sha256": old[qid]["prompt_sha256"],
                "context_ids": old[qid]["rerank"]["Jev"]["context_ids"],
            }
        rows.append(row)
    reranking = {}
    for mode in ("batch", "pointwise"):
        reranking[mode] = {
            "questions": 1044,
            "hits": sum(
                bool(set(r["reranking"][mode]["context_ids"]) & set(qrels[r["query_id"]]))
                for r in rows
            ),
            **{
                name: mean(
                    fn(r["reranking"][mode]["context_ids"], set(qrels[r["query_id"]]), 5)
                    for r in rows
                )
                for name, fn in (
                    ("recall_at_5", recall_at_k),
                    ("ndcg_at_5", ndcg_at_k),
                    ("mrr_at_5", mrr_at_k),
                )
            },
            "cost_usd": sum(r["reranking"][mode]["cost_usd"] for r in rows),
        }
    generation = {}
    for mode in ("Luna", "Haiku"):
        entries = [r["generation"][mode] for r in rows]
        generation[mode] = {
            "questions": 1044,
            "f1": mean(g["f1"] for g in entries),
            "em": mean(g["em"] for g in entries),
            "f1_at_least_half": mean(g["f1"] >= 0.5 for g in entries),
            "failures": sum(g["quality_failure"] for g in entries),
            "missing_usage_responses": sum(g["prompt_tokens"] is None for g in entries),
            "cost_reserve_usd": sum(g["cost_usd"] for g in entries if g["cost_is_reserve"]),
            "cost_usd": sum(g["cost_usd"] for g in entries),
            "prompt_tokens": sum(g["prompt_tokens"] or 0 for g in entries),
            "completion_tokens": sum(g["completion_tokens"] or 0 for g in entries),
            "f1_difference_vs_Gemini": mean(
                r["generation"][mode]["f1"] - old[r["query_id"]]["generation"]["Gemini"]["f1"]
                for r in rows
            ),
            "f1_difference_vs_DeepSeek": mean(
                r["generation"][mode]["f1"] - old[r["query_id"]]["generation"]["DeepSeek"]["f1"]
                for r in rows
            ),
        }
    charged = sum(r["charged_or_reserved_usd"] for r in calls)
    return rows, {
        "reranking": reranking,
        "generation": generation,
        "attempts": len(calls),
        "failed_attempts": sum(not r["ok"] for r in calls),
        "charged_or_reserved_usd": charged,
        "project_total_conservative_usd": 6.565601544360016 + charged,
        "project_cap_usd": 11,
        "limitations": [
            "Same reused 1044-query evaluation set; 240 corpus passages.",
            "Same generator context/prompt and nominal medium effort; effort is not equivalent compute across models.",
            "Temperature unsupported on both new generators; Haiku lacks seed.",
            "Generator responses return undated model names; endpoint metadata snapshots retain advertised dated versions.",
            "No new retrieval or cross-product of every reranker with every generator.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(analyze()[1], indent=2))
