"""Matched Luna Decisions reranking and Luna/Haiku generation; dry run by default."""

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
from jev_rag_benchmark.generation import build_prompt
from jev_rag_benchmark.models import Candidate, Document
from jev_rag_benchmark.rerankers import RELEVANCE_CRITERIA, RELEVANCE_INSTRUCTIONS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/luna-haiku"
LIMIT = 4.40
MODELS = {
    "batch": ("openai/gpt-6-luna-decisions", "OpenAI", "openai/gpt-6-luna-decisions-20261006"),
    "pointwise": ("openai/gpt-6-luna-decisions", "OpenAI", "openai/gpt-6-luna-decisions-20261006"),
    "Luna": ("openai/gpt-6-luna", "OpenAI", "openai/gpt-6-luna-20260922"),
    "Haiku": ("anthropic/claude-haiku-5.5", "Anthropic", "anthropic/claude-haiku-5.5-20261007"),
}


def body(row, mode, index, original):
    model, provider, _ = MODELS[mode]
    request = {"model": model, "provider": {"only": [provider], "allow_fallbacks": False}}
    if mode in ("Luna", "Haiku"):
        docs = {d["doc_id"]: Document(**d) for d in row["candidates"]}
        contexts = [
            Candidate(docs[d], 0, i + 1)
            for i, d in enumerate(original["rerank"]["Jev"]["context_ids"])
        ]
        prompt = build_prompt(
            row["query"], contexts, "Kaynaklarda bu soruyu yanıtlamak için yeterli kanıt yok.", 6000
        )
        assert fingerprint(prompt) == original["prompt_sha256"]
        request.update(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "reasoning": {"effort": "medium"},
            }
        )
        # Temperature is unsupported on both; Haiku also does not support seed.
        if mode == "Luna":
            request.update(
                {
                    "seed": 20260919,
                    "service_tier": "default",
                    "prompt_cache_options": {"mode": "explicit"},
                }
            )
        return request
    if mode == "batch":
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
    request.update({"state": state, "questions": questions})
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    pool = json.loads(
        gzip.decompress((ROOT / "results/fair-2026-10-29/frozen-pool.json.gz").read_bytes())
    )
    originals = {
        r["query_id"]: r
        for r in map(
            json.loads, (ROOT / "results/fair-2026-10-29/results.jsonl").read_text().splitlines()
        )
    }
    protocol = {
        "pool": fingerprint(pool),
        "models": MODELS,
        "limit_usd": LIMIT,
        "requests": {
            mode: body(
                pool[0], mode, 0 if mode == "pointwise" else None, originals[pool[0]["query_id"]]
            )
            for mode in MODELS
        },
        "candidate_k": 20,
        "context_k": 5,
        "tie_break": "original candidate order",
        "rate_per_second": 40,
        "concurrency": 64,
        "transport_retries": 2,
        "generation": "Same original Jev context and prompt; medium reasoning; 4096 output tokens; empty/truncated answer scores zero without quality retries.",
        "parameter_limitations": "Temperature unsupported for Luna/Haiku; seed supported only for Luna. No tools or web search. Luna standard service tier and explicit/no-breakpoint cache mode.",
    }
    protocol = json.loads(json.dumps(protocol))
    digest = fingerprint(protocol)
    if not args.run:
        print("Dry run: 21924 decisions + 2088 generator requests; incremental cap $4.40")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        path.write_text(json.dumps(protocol, indent=2))
    journal = OUT / "calls.jsonl"
    calls = (
        [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    )
    assert all(r["protocol"] == digest for r in calls)
    saved = {(r["query_id"], r["mode"], r["index"]): r for r in calls if r["ok"]}
    spent = sum(r["charged_or_reserved_usd"] for r in calls)
    reserved = 0.0
    next_time = 0.0
    blocked_until = 0.0
    lock = threading.Lock()
    rate_lock = threading.Lock()
    stopped = threading.Event()
    client = httpx.Client(
        timeout=90, limits=httpx.Limits(max_connections=64, max_keepalive_connections=64)
    )
    key = os.environ["OPENROUTER_API_KEY"]

    def pace():
        nonlocal next_time
        while True:
            with rate_lock:
                now = time.monotonic()
                delay = max(next_time, blocked_until) - now
                if delay <= 0:
                    next_time = now + 1 / 40
                    return
            time.sleep(min(delay, 1))

    def execute(task):
        nonlocal spent, reserved, blocked_until
        row, mode, index = task
        identity = (row["query_id"], mode, index)
        if identity in saved:
            return
        request = body(row, mode, index, originals[row["query_id"]])
        generation = mode in ("Luna", "Haiku")
        reserve = 0.004 if generation else 0.006 if mode == "batch" else 0.0015
        endpoint = (
            "https://openrouter.ai/api/v1/chat/completions"
            if generation
            else "https://openrouter.ai/api/alpha/decisions"
        )
        for attempt in range(3):
            pace()
            with lock:
                if stopped.is_set():
                    return
                if spent + reserved + reserve > LIMIT:
                    stopped.set()
                    raise RuntimeError("Budget exhausted")
                reserved += reserve
            record = {
                "query_id": row["query_id"],
                "mode": mode,
                "index": index,
                "protocol": digest,
                "request_sha256": fingerprint(request),
                "attempt": attempt,
                "ok": False,
                "charged_or_reserved_usd": reserve,
            }
            retryable = False
            started = time.perf_counter()
            try:
                response = client.post(
                    endpoint,
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
                    headers = payload.get("error", {}).get("metadata", {}).get("headers", {})
                    try:
                        delay = max(
                            1,
                            min(120, float(headers["X-RateLimit-Reset"]) / 1000 - time.time() + 1),
                        )
                    except (ValueError, KeyError, TypeError):
                        delay = min(60, float(response.headers.get("retry-after", 5)))
                    with rate_lock:
                        blocked_until = max(blocked_until, time.monotonic() + delay)
                retryable = response.status_code == 429 or response.status_code >= 500
                response.raise_for_status()
                assert payload["provider"] == MODELS[mode][1]
                assert payload["model"] in (MODELS[mode][0], MODELS[mode][2])
                if generation:
                    choice = payload["choices"][0]
                    content = choice["message"].get("content") or ""
                    assert isinstance(content, str)
                    record["answer"] = content.strip()
                    record["finish_reason"] = choice.get("finish_reason")
                    record["quality_failure"] = (
                        not record["answer"] or choice.get("finish_reason") == "length"
                    )
                else:
                    scores = (
                        [float(payload["answers"][f"candidate_{i}"]["noul"]) for i in range(20)]
                        if mode == "batch"
                        else [float(payload["answers"]["relevant"]["noul"])]
                    )
                    assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
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
                    reserved -= reserve
                    with journal.open("a") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if record["ok"]:
                        saved[identity] = record
                    (OUT / "status.json").write_text(
                        json.dumps(
                            {
                                "completed": len(saved),
                                "total": 24012,
                                "spent_usd": spent,
                                "limit_usd": LIMIT,
                                "stopped": stopped.is_set(),
                            }
                        )
                    )
                    if len(saved) % 500 == 0:
                        print(f"{len(saved)}/24012; ${spent:.6f}", flush=True)
            if record["ok"]:
                return
            if not retryable or attempt == 2:
                stopped.set()
                raise RuntimeError(f"{identity}: {record.get('error')}")
            time.sleep(2**attempt)

    tasks = [
        (r, mode, index)
        for r in pool
        for mode, index in [
            ("batch", None),
            *[("pointwise", i) for i in range(20)],
            ("Luna", None),
            ("Haiku", None),
        ]
    ]
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(execute, tasks[:230]))
        print(f"Pilot complete: 230 requests; ${spent:.6f}", flush=True)
        if not args.pilot_only:
            with ThreadPoolExecutor(max_workers=64) as executor:
                list(executor.map(execute, tasks[230:]))
            assert len(saved) == 24012
            print(f"COMPLETE ${spent:.6f}", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
