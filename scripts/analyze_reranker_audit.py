"""Check Jev/Cohere score mappings and controlled reranker probes, without API calls."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/reranker-audit-2026-10-29"


def analyze():
    original = {
        r["query_id"]: r
        for r in map(
            json.loads, (ROOT / "results/fair-2026-10-29/results.jsonl").read_text().splitlines()
        )
    }
    pool = {
        r["query_id"]: r
        for r in json.loads(
            gzip.decompress((ROOT / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
        )
    }
    qrels = json.loads((ROOT / "results/fair-2026-10-29/qrels.json").read_text())
    selection = json.loads((DATA / "selection.json").read_text())
    calls = [
        json.loads(s)
        for s in gzip.decompress((DATA / "calls.jsonl.gz").read_bytes()).decode().splitlines()
    ]
    for row in original.values():
        for name in ("Jev", "Cohere"):
            scores = row["rerank"][name]["details"]["scores"]
            assert set(scores) == set(row["candidate_ids"])
            order = sorted(scores, key=lambda d: (-scores[d], row["candidate_ids"].index(d)))
            assert order[:5] == row["rerank"][name]["context_ids"]
    expected = (
        selection["random_ids"]
        + selection["targeted_jev_miss_ids"]
        + selection["targeted_cohere_miss_ids"]
    )
    assert len(expected) == len(set(expected)) == 18
    table = []
    for qid in expected:
        frozen = pool[qid]
        ids = [d["doc_id"] for d in frozen["candidates"]]
        gold = set(qrels[qid])
        item = {
            "query_id": qid,
            "query": frozen["query"],
            "group": "random" if qid in selection["random_ids"] else "targeted",
            "ranks": {},
            "top5": {},
        }

        def add(name, scores):
            assert set(scores) == set(ids)
            ordered = sorted(ids, key=lambda d: (-scores[d], ids.index(d)))
            item["ranks"][name] = next((i + 1 for i, d in enumerate(ordered) if d in gold), None)
            item["top5"][name] = ordered[:5]

        for name in ("Jev", "Cohere"):
            add(name + "_original", original[qid]["rerank"][name]["details"]["scores"])
        pointwise = {}
        found = set()
        for record in calls:
            if record["query_id"] != qid:
                continue
            variant = record["variant"]
            request, response = record["request"], record["response"]
            assert record["status"] == 200
            if variant == "jev_pointwise":
                i = record["index"]
                assert i not in pointwise
                assert request["state"]["candidate"] == {
                    "id": ids[i],
                    "text": frozen["candidates"][i]["text"],
                }
                assert request["state"]["query"] == frozen["query"]
                pointwise[i] = response["answers"]["relevant"]["noul"]
                continue
            assert variant not in found
            found.add(variant)
            reverse = variant in ("jev_reverse_candidates", "cohere_reverse")
            ds = list(reversed(frozen["candidates"])) if reverse else frozen["candidates"]
            if variant.startswith("cohere"):
                assert request["documents"] == [d["text"] for d in ds]
                assert request["query"] == frozen["query"]
                entries = response["results"]
                assert len(entries) == 20 and {r["index"] for r in entries} == set(range(20))
                scores = {ds[r["index"]]["doc_id"]: r["relevance_score"] for r in entries}
            else:
                assert request["state"]["candidates"] == [
                    {"id": d["doc_id"], "text": d["text"]} for d in ds
                ]
                assert request["state"]["query"] == frozen["query"]
                scores = {
                    d["doc_id"]: response["answers"][f"candidate_{i}"]["noul"]
                    for i, d in enumerate(ds)
                }
            add(variant, scores)
        assert len(found) == 6 and set(pointwise) == set(range(20))
        add("jev_pointwise", {ids[i]: value for i, value in pointwise.items()})
        table.append(item)
    variants = list(table[0]["ranks"])
    groups = {
        group: {v: sum(r["ranks"][v] <= 5 for r in table if r["group"] == group) for v in variants}
        for group in ("random", "targeted")
    }
    overlaps = {}
    for model, repeat, reverse in (
        ("Jev", "jev_repeat", "jev_reverse_candidates"),
        ("Cohere", "cohere_repeat", "cohere_reverse"),
    ):
        overlaps[model] = {
            "top5_set_changed_on_repeat": sum(
                set(r["top5"][model + "_original"]) != set(r["top5"][repeat]) for r in table
            ),
            "gold_rank_changed_on_repeat": sum(
                r["ranks"][model + "_original"] != r["ranks"][repeat] for r in table
            ),
            "top5_set_changed_on_reverse": sum(
                set(r["top5"][repeat]) != set(r["top5"][reverse]) for r in table
            ),
            "gold_rank_changed_on_reverse": sum(
                r["ranks"][repeat] != r["ranks"][reverse] for r in table
            ),
        }
    return {
        "full_set_score_mapping_checks": {
            "queries": 1044,
            "Jev_mismatches": 0,
            "Cohere_mismatches": 0,
        },
        "selection": selection,
        "questions": table,
        "top5_hits": groups,
        "order_sensitivity": overlaps,
        "calls": len(calls),
        "cost_usd": sum(r["cost_or_reserve"] for r in calls),
        "limits": "16 random gold-present cases plus 2 targeted failures, not proof of general invariance; original rank used to break ties.",
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
