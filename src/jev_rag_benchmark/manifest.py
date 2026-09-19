from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import sha256_file


TRACKED_PACKAGES = ["httpx", "numpy", "pydantic", "PyYAML", "typer", "typesafe-sdk"]


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_versions() -> dict[str, str]:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def build_manifest(
    root: Path,
    config: dict[str, Any],
    dataset_paths: list[Path],
    *,
    run_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_kind": run_kind,
        "git_commit": git_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": package_versions(),
        "dataset_files": {
            str(path.relative_to(root)): sha256_file(path) for path in dataset_paths if path.exists()
        },
        "models": {
            "generator_requested": config["generator"]["model"],
            "jev_requested": config["rerankers"]["jev"]["model"],
            "cross_encoder_requested": config["rerankers"]["cross_encoder"]["model"],
        },
        "prices": {
            "jev_input_usd_per_million_tokens": config["rerankers"]["jev"][
                "input_usd_per_million_tokens"
            ],
            "verified_on": str(config["rerankers"]["jev"]["price_verified_on"]),
        },
        "seed": config["run"]["seed"],
        "prompt_versions": {
            "answer": config["generator"]["prompt_version"],
            "jev": config["rerankers"]["jev"]["prompt_version"],
        },
        "cache": config["run"]["cache"],
        "concurrency": config["run"]["concurrency"],
        "config": config,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
