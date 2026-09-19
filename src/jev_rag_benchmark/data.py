from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .io import write_jsonl

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
XQUAD_URL = "https://raw.githubusercontent.com/google-deepmind/xquad/master/xquad.{lang}.json"


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def prepare_scifact(root: Path, *, force: bool = False) -> tuple[Path, Path]:
    raw_zip = root / "data/raw/scifact.zip"
    extracted = root / "data/raw/scifact"
    if force or not raw_zip.exists():
        _download(SCIFACT_URL, raw_zip)
    if force or not extracted.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw_zip) as archive:
            archive.extractall(root / "data/raw")

    corpus_path = extracted / "corpus.jsonl"
    query_path = extracted / "queries.jsonl"
    qrels_path = extracted / "qrels/test.tsv"
    documents = []
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            documents.append(
                {
                    "doc_id": str(row["_id"]),
                    "title": row.get("title", ""),
                    "text": row["text"],
                    "metadata": row.get("metadata", {}),
                }
            )
    qrels: dict[str, list[str]] = {}
    with qrels_path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            query_id, doc_id, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(query_id, []).append(doc_id)
    queries = []
    with query_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = str(row["_id"])
            if query_id in qrels:
                queries.append(
                    {
                        "query_id": query_id,
                        "text": row["text"],
                        "relevant_doc_ids": qrels[query_id],
                        "answers": [],
                        "language": "en",
                        "split": "test",
                    }
                )
    docs_out = root / "data/processed/scifact/documents.jsonl"
    queries_out = root / "data/processed/scifact/queries.test.jsonl"
    write_jsonl(docs_out, documents)
    write_jsonl(queries_out, queries)
    return docs_out, queries_out


def _stable_split(article_title: str) -> str:
    value = int(hashlib.sha256(article_title.encode()).hexdigest()[:8], 16) % 10
    return "dev" if value == 0 else "test"


def prepare_xquad(root: Path, lang: str, *, force: bool = False) -> tuple[Path, Path]:
    raw = root / f"data/raw/xquad.{lang}.json"
    if force or not raw.exists():
        _download(XQUAD_URL.format(lang=lang), raw)
    payload = json.loads(raw.read_text(encoding="utf-8"))
    documents: dict[str, dict] = {}
    queries = []
    for article in payload["data"]:
        split = _stable_split(article["title"])
        for para_idx, paragraph in enumerate(article["paragraphs"]):
            doc_id = f"xquad-{lang}-{hashlib.sha256((article['title'] + str(para_idx)).encode()).hexdigest()[:12]}"
            documents[doc_id] = {
                "doc_id": doc_id,
                "title": article["title"],
                "text": paragraph["context"],
                "metadata": {"source": "XQuAD", "language": lang},
            }
            for qa in paragraph["qas"]:
                queries.append(
                    {
                        "query_id": f"xquad-{lang}-{qa['id']}",
                        "text": qa["question"],
                        "relevant_doc_ids": [doc_id],
                        "answers": sorted({answer["text"] for answer in qa["answers"]}),
                        "language": lang,
                        "split": split,
                    }
                )
    base = root / f"data/processed/xquad-{lang}"
    docs_out = base / "documents.jsonl"
    queries_out = base / "queries.jsonl"
    write_jsonl(docs_out, documents.values())
    write_jsonl(queries_out, queries)
    return docs_out, queries_out


def prepare_all(root: Path, *, force: bool = False) -> list[Path]:
    paths = []
    paths.extend(prepare_scifact(root, force=force))
    paths.extend(prepare_xquad(root, "en", force=force))
    paths.extend(prepare_xquad(root, "tr", force=force))
    return paths

