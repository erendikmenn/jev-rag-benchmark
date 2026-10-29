"""Immutable, content-addressed retrieval snapshots and comparison validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .models import Candidate, Document


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def freeze_candidates(path: Path, documents, queries, retrieval_config, build_index):
    """Reuse only an exactly matching snapshot; never silently overwrite one."""
    identity = fingerprint(
        {
            "documents": [asdict(d) for d in documents],
            "queries": [asdict(q) for q in queries],
            "retrieval": retrieval_config,
        }
    )
    if path.exists():
        payload = json.loads(path.read_text())
        if payload["identity"] != identity:
            raise ValueError("frozen candidate pool does not match dataset or retrieval settings")
        if payload["digest"] != fingerprint(payload["queries"]):
            raise ValueError("frozen candidate pool content was modified")
    else:
        index = build_index()
        entries = {}
        for query in queries:
            entries[query.query_id] = [
                asdict(c) for c in index.retrieve(query.text, retrieval_config["candidate_k"])
            ]
        payload = {"identity": identity, "queries": entries, "digest": fingerprint(entries)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    pools = {
        qid: [Candidate(**{**c, "document": Document(**c["document"])}) for c in cs]
        for qid, cs in payload["queries"].items()
    }
    return pools, payload["digest"]


def validate_comparison(rows):
    """Reject duplicate observations and mixed protocols before publishing."""
    if not rows:
        raise ValueError("no result rows")
    protocols = {r.get("protocol_id") for r in rows}
    if None in protocols or len(protocols) != 1:
        raise ValueError(
            "missing or incompatible benchmark protocol_id; legacy results require a separate audit"
        )
    seen = set()
    by_query = {}
    coverage = {}
    for row in rows:
        key = (row["query_id"], row["branch"])
        if key in seen:
            raise ValueError("duplicate query/branch observation")
        seen.add(key)
        coverage.setdefault(row["branch"], set()).add(row["query_id"])
        signature = (row.get("query"), row.get("candidate_ids"), row.get("references"))
        if row["query_id"] in by_query and by_query[row["query_id"]] != signature:
            raise ValueError("candidate order, question or references differ between branches")
        by_query[row["query_id"]] = signature
    if any(ids != next(iter(coverage.values())) for ids in coverage.values()):
        raise ValueError("branches have unequal query coverage")
