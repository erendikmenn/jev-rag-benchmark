"""Audit saved diagnostic probes; no paid calls."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/clef-flash-diagnostics-2026-10-29"


def analyze():
    selection = json.loads((DATA / "selection.json").read_text())
    pool = {
        r["query_id"]: r
        for r in json.loads(
            gzip.decompress((ROOT / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
        )
    }
    qrels = json.loads((ROOT / "results/fair-2026-10-29/qrels.json").read_text())
    records = [
        json.loads(line)
        for name in ("probes.jsonl", "followup.jsonl")
        for line in (DATA / name).read_text().splitlines()
    ]
    expected = selection["failure_ids"] + selection["success_ids"]
    assert len(set(expected)) == 12
    table = []
    pointwise_cost = 0
    repeat_cost = 0
    for qid in expected:
        frozen = pool[qid]
        docs = frozen["candidates"]
        gold = set(qrels[qid])
        gi = next(i for i, doc in enumerate(docs) if doc["doc_id"] in gold)
        item = {
            "query_id": qid,
            "query": frozen["query"],
            "group": "original_failure" if qid in selection["failure_ids"] else "control",
            "reference_answers": frozen["references"],
            "original_candidate_rank": gi + 1,
        }
        scores = {}
        for r in records:
            if r["query_id"] != qid:
                continue
            request = r["request"]
            response = r["response"]
            assert request["model"] == response["model"] == "cloudflare/clef-flash"
            assert request["provider"]["only"] == [response["provider"]]
            assert request["provider"]["allow_fallbacks"] is False
            assert request["state"]["query"] == frozen["query"]
            variant = r["variant"]
            if variant in ("pointwise", "pointwise_pair"):
                i = r["candidate_index"]
                assert i not in scores
                assert request["state"]["candidate"] == {
                    "id": docs[i]["doc_id"],
                    "text": docs[i]["text"],
                }
                scores[i] = response["answers"]["relevant"]["noul"]
                pointwise_cost += response["usage"]["cost"]
            elif variant != "gold_alone":
                candidates = request["state"]["candidates"]
                expected_docs = list(reversed(docs)) if variant == "reverse_candidates" else docs
                assert candidates == [{"id": d["doc_id"], "text": d["text"]} for d in expected_docs]
                vals = [response["answers"][f"candidate_{i}"]["noul"] for i in range(20)]
                order = sorted(range(20), key=lambda i: (-vals[i], i))
                gold_index = next(i for i, d in enumerate(candidates) if d["id"] in gold)
                item[variant + "_gold_rank"] = order.index(gold_index) + 1
                if variant == "repeat":
                    repeat_cost += response["usage"]["cost"]
        assert set(scores) == set(range(20))
        assert all(0 <= score <= 1 for score in scores.values())
        order = sorted(scores, key=lambda i: (-scores[i], i))
        item["pointwise_gold_rank"] = order.index(gi) + 1
        table.append(item)
    costs = sum(r["response"]["usage"]["cost"] for r in records)
    counts = {
        variant: {
            group: sum(r[variant + "_gold_rank"] <= 5 for r in table if r["group"] == group)
            for group in ("original_failure", "control")
        }
        for variant in (
            "repeat",
            "explicit",
            "reverse_questions",
            "reverse_candidates",
            "provider_bf16",
            "pointwise",
        )
    }
    return {
        "selection_warning": "Six outcome-selected failures and six successful controls; NOT a new unbiased benchmark.",
        "questions": table,
        "top5_hits_out_of_6_per_group": counts,
        "paid_calls": len(records),
        "cost_usd": costs,
        "pointwise_240_calls_cost_usd": pointwise_cost,
        "repeat_12_batch_calls_cost_usd": repeat_cost,
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
