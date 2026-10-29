"""Frozen full-set Jev 1.14 batch and pointwise run. Dry run unless --run is supplied."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from jev_rag_benchmark.fairness import fingerprint
from jev_rag_benchmark.rerankers import RELEVANCE_CRITERIA, RELEVANCE_INSTRUCTIONS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/jev114"
LIMIT = 1.45
RESERVE = 0.0005
MODEL = "typesafe/jev-1.14"
RESOLVED_MODEL = "typesafe/jev-1.14-20261014"
PROVIDER = "TypeSafe"


def body(row, index):
    if index is None:
        state = {
            "query": row["query"],
            "candidates": [{"id": d["doc_id"], "text": d["text"]} for d in row["candidates"]],
        }
        questions = {
            f"candidate_{i}": {
                "type": "noul",
                "instructions": RELEVANCE_INSTRUCTIONS + f" Evaluate only `candidates[{i}]`.",
                "criteria": RELEVANCE_CRITERIA,
            }
            for i in range(20)
        }
    else:
        doc = row["candidates"][index]
        state = {"query": row["query"], "candidate": {"id": doc["doc_id"], "text": doc["text"]}}
        questions = {
            "relevant": {
                "type": "noul",
                "instructions": RELEVANCE_INSTRUCTIONS,
                "criteria": RELEVANCE_CRITERIA,
            }
        }
    return {
        "model": MODEL,
        "provider": {"only": [PROVIDER], "allow_fallbacks": False},
        "state": state,
        "questions": questions,
    }


def rate_limit_delay(payload, now):
    headers = payload.get("error", {}).get("metadata", {}).get("headers", {})
    try:
        reset = float(headers["X-RateLimit-Reset"]) / 1000
    except (KeyError, TypeError, ValueError):
        return 5.0
    return max(1.0, min(120.0, reset - now + 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    pool = json.loads(
        gzip.decompress((ROOT / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
    )
    protocol = {
        "model": MODEL,
        "provider": PROVIDER,
        "pool": fingerprint(pool),
        "example_requests": [body(pool[0], None), body(pool[0], 0)],
        "resolved_model": RESOLVED_MODEL,
        "top_k": 5,
        "tie_break": "original candidate order",
        "concurrency": 64,
        "max_transport_retries": 2,
        "limit_usd": LIMIT,
        "note": "Both modes on all 1044 frozen queries; batch prompt identical to original Jev 1.13, pointwise prompt identical to Clef Flash.",
    }
    digest = fingerprint(protocol)
    if not args.run:
        print(f"Dry run: {len(pool) * 21} calls; incremental cap ${LIMIT}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "protocol.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == protocol
    else:
        manifest.write_text(json.dumps(protocol, indent=2))
    journal = OUT / "calls.jsonl"
    calls = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
    assert all(r["protocol"] == digest for r in calls)
    saved = {(r["query_id"], r["index"]): r for r in calls if r["ok"]}
    pilot = (
        [json.loads(line) for line in (OUT / "pilot.jsonl").read_text().splitlines()]
        if (OUT / "pilot.jsonl").exists()
        else []
    )
    spent = sum(r["charged_or_reserved_usd"] for r in calls + pilot)
    reserved = 0.0
    lock = threading.Lock()
    stopped = threading.Event()
    rate_lock = threading.Lock()
    next_request = 0.0
    blocked_until = 0.0

    def pace():
        nonlocal next_request
        while True:
            with rate_lock:
                now = time.monotonic()
                wait = max(blocked_until, next_request) - now
                if wait <= 0:
                    next_request = now + 1 / 25
                    return
            time.sleep(min(wait, 1.0))

    key = os.environ["OPENROUTER_API_KEY"]
    client = httpx.Client(
        timeout=60, limits=httpx.Limits(max_connections=64, max_keepalive_connections=64)
    )

    def execute(task):
        nonlocal reserved, spent, blocked_until
        row, index = task
        identity = (row["query_id"], index)
        if identity in saved:
            return
        request = body(row, index)
        reservation = 0.002 if index is None else RESERVE
        for attempt in range(3):
            pace()
            with lock:
                if stopped.is_set():
                    return
                if spent + reserved + reservation > LIMIT:
                    stopped.set()
                    raise RuntimeError("budget exhausted")
                reserved += reservation
            record = {
                "query_id": row["query_id"],
                "index": index,
                "protocol": digest,
                "request_sha256": fingerprint(request),
                "attempt": attempt,
                "ok": False,
                "charged_or_reserved_usd": reservation,
            }
            retryable = False
            started = time.perf_counter()
            try:
                response = client.post(
                    "https://openrouter.ai/api/alpha/decisions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "X-OpenRouter-Title": "jev-rag-benchmark",
                    },
                    json=request,
                )
                record["status"] = response.status_code
                payload = response.json()
                record["response"] = payload
                usage = payload.get("usage", {})
                if "cost" in usage:
                    record["charged_or_reserved_usd"] = float(usage["cost"])
                elif response.status_code == 429:
                    record["charged_or_reserved_usd"] = 0
                if response.status_code == 429:
                    delay = rate_limit_delay(payload, time.time())
                    with rate_lock:
                        blocked_until = max(blocked_until, time.monotonic() + delay)
                    record["rate_limit_wait_seconds"] = delay
                retryable = response.status_code == 429 or response.status_code >= 500
                response.raise_for_status()
                assert payload["model"] == RESOLVED_MODEL and payload["provider"] == PROVIDER
                scores = (
                    [float(payload["answers"][f"candidate_{i}"]["noul"]) for i in range(20)]
                    if index is None
                    else [float(payload["answers"]["relevant"]["noul"])]
                )
                assert all(math.isfinite(score) and 0 <= score <= 1 for score in scores)
                assert 0 < usage["input_tokens"] < 32000
                record["scores"] = scores
                record["ok"] = True
            except Exception as exc:
                record["error"] = str(exc)
                retryable = retryable or isinstance(exc, httpx.TransportError)
            finally:
                record["latency_ms"] = (time.perf_counter() - started) * 1000
                with lock:
                    spent += record["charged_or_reserved_usd"]
                    reserved -= reservation
                    with journal.open("a") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if record["ok"]:
                        saved[identity] = record
                    (OUT / "status.json").write_text(
                        json.dumps(
                            {
                                "completed_calls": len(saved),
                                "total_calls": 21924,
                                "spent_usd": spent,
                                "limit_usd": LIMIT,
                                "stopped": stopped.is_set(),
                            }
                        )
                    )
                    if len(saved) % 500 == 0:
                        print(f"{len(saved)}/21924 calls; ${spent:.6f}", flush=True)
            if record["ok"]:
                return
            if not retryable or attempt == 2:
                stopped.set()
                raise RuntimeError(f"{identity}: {record.get('error')}")
            time.sleep(2**attempt)

    tasks = [(row, i) for row in pool for i in [None, *range(20)]]
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(execute, tasks[:210]))
        print(f"Pilot: 210 calls completed; ${spent:.6f}", flush=True)
        if not args.pilot_only:
            with ThreadPoolExecutor(max_workers=64) as executor:
                list(executor.map(execute, tasks[210:]))
            assert len(saved) == 21924
            print(f"COMPLETE ${spent:.6f}", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
