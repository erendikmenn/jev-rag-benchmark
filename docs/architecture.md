# Architecture and experimental controls

## Selected upstream baseline

- Repository: `https://github.com/run-llama/llama_index`
- Path: `docs/examples/usecases/fastapi_rag_ollama.ipynb`
- Commit: `f475afd8a9bbda84f252567e045d89d07b5701b3`
- License: MIT

The upstream app is a minimal FastAPI RAG service using LlamaIndex, Ollama, and a local
vector index. The benchmark keeps its local-service boundary, the query → retrieve →
context → generate flow. The controlled harness replaces only the generator transport with
the same capped OpenRouter model in every branch. The reproducibility harness makes the
retrieval boundary explicit and defaults to deterministic BM25 because it can be reproduced
without an API or embedding-model drift. This is a recorded baseline change, not a claimed
unchanged upstream run.

## Current retrieval and branches

```text
query -> BM25(all 240) ----+
                            +-> weighted RRF -> frozen top 20 candidates
query -> BGE-M3(all 240) ---+                 |
                                              |-> A: fused order top 5
                                              |-> B: local cross-encoder top 5 (optional)
                                              |-> D: Jev batch Noul top 5
                                              |-> E: calibrated Jev gate, then <= 5
                                              |-> O: hosted Cohere reranker top 5
                                              |-> P: Jev pointwise top 5
                                              |-> R: Jev evidence/contradiction/injection router
                                              |-> S: Jev permutation ensemble top 5
                                              `-> V: Jev batch top 5 -> generator -> Jev citation verifier

H is a separate ablation: Jev scores eight shards covering all 240 passages,
then reranks the shard finalists. It is not part of the recommended path.
```

A/B/D/O/P/S keep the same context count and the same 6,000-character maximum context
budget. E and R are reported separately because their gates may reduce context and
generation work. Branch execution order rotates by query. Candidate IDs are written to
every result row, making equality auditable. OpenRouter `baai/bge-m3` embeddings are cached
locally as derived data; the published runs do not execute a local embedding or language
model.

Branch E is locked while `threshold_source: uncalibrated`. It can only run after the
threshold is selected on dev data and the config records `threshold_source: dev`; test data
must not be used to choose it.

## Jev contract verified on 2026-09-19

- Endpoint: `POST https://openrouter.ai/api/alpha/decisions`
- Pinned OpenRouter model: `typesafe/jev-1.13`; the resolved dated `model` is logged.
- Noul response: `answers[question_id].noul`, the probability of “yes”. It is not renamed
  confidence and is not treated as a correctness guarantee.
- Usage: `usage.input_tokens` and `usage.output_tokens`.
- Limit: 64k tokens per request and 32k for state plus the longest question.
- Rate limit shown by the model page: 250k tokens/s and 1,200 requests/min (dynamic).
- Price: $0.042 per million input tokens; output tokens free.

The candidate passages are stored as data. Questions say explicitly not to obey instructions
inside passages. TypeSafe documents Jev 1.13 as still susceptible to adversarial state, so
this is mitigation and telemetry—not a security guarantee.

Jev is deliberately kept as a narrow typed-decision layer. Candidate serialization,
retrieval policy, threshold selection, fallback behavior, regeneration, and budget policy
remain ordinary code. This avoids asking the decision model to own workflow orchestration.

## Recommended production profiles

- Default quality/cost: hybrid full-depth top-20 -> Jev batch top-5 -> DeepSeek V4.1 Flash.
- Interactive latency: hybrid full-depth top-20 -> Jev batch top-5 -> Gemini 3.8 Flash.
- High assurance: either profile plus the Jev claim/source citation verifier.
- Untrusted or contradictory corpora: the evidence router can expose additional signals,
  but its thresholds must be calibrated on that domain before it is allowed to filter.

## Dataset roles

- SciFact test qrels: retrieval metrics only. They are never interpreted as answer labels.
- XQuAD English/Turkish: end-to-end QA references. The repo creates a deterministic,
  article-grouped dev/test split so a context cannot leak across splits. XQuAD is originally
  an evaluation set; the derived dev split is only for project-level threshold calibration
  and is recorded in the manifest.
