"""Run the frozen Clef Flash comparison; paid calls require --run."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from jev_rag_benchmark.fairness import fingerprint
from jev_rag_benchmark.rerankers import RELEVANCE_CRITERIA, RELEVANCE_INSTRUCTIONS

ROOT = Path(__file__).resolve().parents[1]
MODEL = "cloudflare/clef-flash"
PROVIDER = "DeepInfra"
RATE = 0.019
LIMIT = 0.60
RESERVATION = 0.001  # exceeds the provider's 24,576-token maximum at the pinned rate


def request_body(row):
    return {
        "model": MODEL,
        "provider": {"only": [PROVIDER], "allow_fallbacks": False},
        "state": {
            "query": row["query"],
            "candidates": [{"id": d["doc_id"], "text": d["text"]} for d in row["candidates"]],
        },
        "questions": {
            f"candidate_{i}": {
                "type": "noul",
                "instructions": RELEVANCE_INSTRUCTIONS + f" Evaluate only `candidates[{i}]`.",
                "criteria": RELEVANCE_CRITERIA,
            }
            for i in range(len(row["candidates"]))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    pool = json.loads(
        gzip.decompress((ROOT / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
    )
    digest = fingerprint(pool)
    protocol = fingerprint(
        {
            "pool": digest,
            "example_body": request_body(pool[0]),
            "tie_break": "original candidate order",
            "top_k": 5,
        }
    )
    if not args.run:
        print(f"Dry run: {len(pool)} queries, {MODEL}, {PROVIDER}, cap ${LIMIT:.2f}")
        return
    key = os.environ["OPENROUTER_API_KEY"]
    out = ROOT / "work/clef-flash"
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.jsonl"
    results = (
        [json.loads(s) for s in result_path.read_text().splitlines()]
        if result_path.exists()
        else []
    )
    assert all(r["protocol"] == protocol for r in results), "Cannot resume changed protocol"
    saved = {r["query_id"]: r for r in results}
    assert len(saved) == len(results)
    status_path = out / "status.json"
    initial = json.loads(status_path.read_text()) if status_path.exists() else {}
    preflight = (
        json.loads((out / "preflight.json").read_text())
        if (out / "preflight.json").exists()
        else {}
    )
    spent = initial.get(
        "spent_usd", (preflight.get("response", {}).get("usage", {}).get("cost", 0))
    )
    lock = threading.Lock()
    stopped = threading.Event()
    reserved = 0.0

    def status():
        status_path.write_text(
            json.dumps(
                {
                    "spent_usd": spent,
                    "completed": len(saved),
                    "protocol": protocol,
                    "limit_usd": LIMIT,
                }
            )
        )

    def execute(row):
        nonlocal spent, reserved
        if row["query_id"] in saved:
            return
        with lock:
            if stopped.is_set() or spent + reserved + RESERVATION > LIMIT:
                raise RuntimeError("Stopped or budget exhausted")
            reserved += RESERVATION
        started = time.perf_counter()
        cost = RESERVATION  # retain conservative reserve if response outcome is unknown
        try:
            with httpx.Client(timeout=90) as client:
                response = client.post(
                    "https://openrouter.ai/api/alpha/decisions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "X-OpenRouter-Title": "jev-rag-benchmark",
                    },
                    json=request_body(row),
                )
            payload = response.json()
            usage = payload.get("usage", {})
            if "cost" in usage:
                cost = float(usage["cost"])
            response.raise_for_status()
            assert payload["provider"] == PROVIDER
            assert payload["model"] == MODEL
            assert 0 < usage["input_tokens"] < 24576, "Potential context truncation"
            scores = [
                float(payload["answers"][f"candidate_{i}"]["noul"])
                for i in range(len(row["candidates"]))
            ]
            assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            record = {
                "query_id": row["query_id"],
                "query": row["query"],
                "candidate_ids": [d["doc_id"] for d in row["candidates"]],
                "context_ids": [row["candidates"][i]["doc_id"] for i in order[:5]],
                "pool_sha256": digest,
                "protocol": protocol,
                "request_sha256": fingerprint(request_body(row)),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "response": payload,
            }
            with lock:
                with result_path.open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved[row["query_id"]] = record
        except Exception as exc:
            stopped.set()
            with lock:
                with (out / "errors.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "query_id": row["query_id"],
                                "error": str(exc),
                                "cost_or_reserve": cost,
                            }
                        )
                        + "\n"
                    )
            raise
        finally:
            with lock:
                spent += cost
                reserved -= RESERVATION
                status()
                if len(saved) % 50 == 0:
                    print(f"Completed {len(saved)}/1044; spend ${spent:.6f}", flush=True)

    rows = list(pool)
    random.Random(20261030).shuffle(rows)
    for row in rows[:10]:
        execute(row)
    print(f"Pilot passed: 10 matched requests; cumulative spend ${spent:.6f}", flush=True)
    if args.pilot_only:
        return
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(execute, rows[10:]))
    print(f"COMPLETE: {len(saved)}; spend ${spent:.6f}", flush=True)


if __name__ == "__main__":
    main()
