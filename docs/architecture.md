# Architecture and experimental controls

## Selected upstream baseline

- Repository: `https://github.com/run-llama/llama_index`
- Path: `docs/examples/usecases/fastapi_rag_ollama.ipynb`
- Commit: `f475afd8a9bbda84f252567e045d89d07b5701b3`
- License: MIT

The upstream app is a minimal FastAPI RAG service using LlamaIndex, Ollama, and a local
vector index. The benchmark keeps its local-service boundary, the query → retrieve →
context → generate flow, and its Ollama generator. The reproducibility harness makes the
retrieval boundary explicit and defaults to deterministic BM25 because it can be reproduced
without an API or embedding-model drift. This is a recorded baseline change, not a claimed
unchanged upstream run.

## Branches

```text
query -> BM25 top 20 -> frozen candidate ids -> A: original top 5 ---------+
                                      |-> B: cross-encoder top 5 ---------+-> identical prompt -> Ollama
                                      |-> D: Jev Noul top 5 --------------+
                                      `-> E: Jev Noul threshold then <= 5-+
```

A/B/D keep the same context count and the same 6,000-character maximum context budget. E is reported separately because it may reduce context
and generation work. Branch execution order rotates by query. Candidate ids are written to
every result row, making equality auditable.

Branch E is locked while `threshold_source: uncalibrated`. It can only run after the
threshold is selected on dev data and the config records `threshold_source: dev`; test data
must not be used to choose it.

## Jev contract verified on 2026-09-19

- Endpoint: `POST https://api.typesafe.ai/v1/systemone`
- Pinned model: `jev-1.13.0`; returned `model` is logged.
- Noul response: `answers[question_id].noul`, the probability of “yes”. It is not renamed
  confidence and is not treated as a correctness guarantee.
- Usage: `usage.input_tokens` and `usage.output_tokens`.
- Limit: 64k tokens per request and 32k for state plus the longest question.
- Rate limit shown by the model page: 250k tokens/s and 1,200 requests/min (dynamic).
- Price: $0.042 per million input tokens; output tokens free.

The candidate passages are stored as data. Questions say explicitly not to obey instructions
inside passages. TypeSafe documents Jev 1.13 as still susceptible to adversarial state, so
this is mitigation and telemetry—not a security guarantee.

## Dataset roles

- SciFact test qrels: retrieval metrics only. They are never interpreted as answer labels.
- XQuAD English/Turkish: end-to-end QA references. The repo creates a deterministic,
  article-grouped dev/test split so a context cannot leak across splits. XQuAD is originally
  an evaluation set; the derived dev split is only for project-level threshold calibration
  and is recorded in the manifest.
