# jev-rag-benchmark

Reproducible, vendor-neutral experiments for measuring whether TypeSafe Jev improves a
small RAG system. “Jev wins” is not an assumption: quality, latency, and cost can improve,
stay flat, or get worse.

## Locked XQuAD-TR result (2026-09-20)

The current full run contains 1,044 unique Turkish XQuAD questions. Published model calls
use OpenRouter; no local LLM or local embedding model is used.

| Stage / model | Main result | Observed cost |
|---|---:|---:|
| Hybrid BM25 + BGE-M3 candidate recall@20 | 1,039/1,044 (99.521%) | query embedding included below |
| Hybrid order, Recall@5 / nDCG@10 | 97.605% / 94.314% | $0 reranking |
| Jev 1.13, Recall@5 / nDCG@10 | 99.425% / 98.097% | $0.410866 / 1,044 queries |
| Cohere Rerank 3.5, Recall@5 / nDCG@10 | 99.425% / 98.626% | $1.044000 / 1,044 queries |
| Jev + Gemini 3.8 Flash successful answers | 163/1,044 (15.61%) | $2.868223 total |
| Jev + DeepSeek V4.1 Flash successful answers | 194/1,044 (18.58%) | $1.070110 total |

DeepSeek produced 31 more successful answers and cost 62.7% less end to end than Gemini,
but its observed median generation latency was 7.64 s versus Gemini's 3.13 s. Jev matched
Cohere's Recall@5 at 60.6% lower reranking cost. Retrieval success and answer success are
different metrics; the detailed interpretation, confidence intervals, ablations, and raw
artifact links are in [`docs/benchmark-2026-09-20.md`](docs/benchmark-2026-09-20.md).

The project is based on LlamaIndex's MIT-licensed local FastAPI/Ollama RAG example at
commit `f475afd8a9bbda84f252567e045d89d07b5701b3`; see
[`docs/architecture.md`](docs/architecture.md) and [`NOTICE`](NOTICE).
The capped three-candidate comparison is in
[`docs/candidate-review.md`](docs/candidate-review.md).

## Install

Requirements: macOS/Linux, Python 3.11–3.13, `uv`, and an OpenRouter API key.

```bash
uv sync --extra dev
export OPENROUTER_API_KEY="..."
uv run jev-rag doctor
```

The optional `cross-encoder` extra is only for branch B and was not used in the published
OpenRouter comparison. `.env` is ignored by Git; no credential file is committed.

Optional FastAPI surface, retained from the selected upstream app shape:

```bash
uv run uvicorn jev_rag_benchmark.app:app --host 127.0.0.1 --port 8000
```

Jev is called through OpenRouter's Decisions endpoint with `OPENROUTER_API_KEY`.
Credentials remain server-side/environment-only. The CLI reports whether a key exists but
never prints it.

## Reproduce

```bash
# Download SciFact and XQuAD EN/TR, then normalize without committing raw data
uv run jev-rag data prepare

# Validate/materialize the deterministic index
uv run jev-rag index --dataset scifact

# One sourced baseline question through the capped OpenRouter config
uv run jev-rag ask "Türkiye'nin başkenti neresidir?" --dataset xquad-tr --branch A \
  --config-path configs/openrouter-smoke.yaml

# 25-query infrastructure smoke. Jev fixture is visibly marked; no paid calls.
uv run jev-rag benchmark smoke --dataset scifact --branches A,B,D --fixture-jev --skip-generation

# Preflight cost from the real top-20 candidate texts; makes zero API calls.
uv run jev-rag estimate --dataset scifact --jev-branches 1

# QA smoke with OpenRouter answer generation
uv run jev-rag benchmark full --dataset xquad-tr --branches A,D --limit 5 \
  --no-fixture-jev --no-skip-generation --config-path configs/openrouter-smoke.yaml

# Audit the first-stage ceiling, then run the locked full A/Jev/Cohere comparison.
uv run jev-rag retrieval-audit --dataset xquad-tr --candidate-ks 5,20,50,100,240 \
  --config-path configs/openrouter-hybrid-fullfusion-20.yaml
uv run jev-rag benchmark full --dataset xquad-tr --branches A,D,O \
  --no-fixture-jev --skip-generation \
  --config-path configs/openrouter-hybrid-fullfusion-20.yaml
uv run jev-rag report results/xquad-tr-a-d-o.jsonl \
  --output-dir reports/generated/xquad-tr-a-d-o

# Replay a generator over frozen Jev contexts. The command checkpoints every ten rows,
# resumes successful rows, retries empty OpenRouter generations, and supports sharding.
uv run python scripts/replay_generator.py \
  results/xquad-tr-hybrid-fullfusion20-a-d-o.jsonl \
  data/processed/xquad-tr/documents.jsonl \
  configs/openrouter-gemini-3.8-hybrid.yaml results/gemini-replay.jsonl \
  --source-branch D --target-branch G --concurrency 4

# Optional dev-only threshold calibration and candidate-order stability audit.
uv run jev-rag calibrate-jev --dataset xquad-tr --minimum-recall 0.95
uv run jev-rag jev-stability --dataset xquad-tr --limit 50 --permutations 5 \
  --config-path configs/openrouter-hybrid-fullfusion-20.yaml

# Real 25-query Jev smoke through OpenRouter, capped at $0.02.
uv run jev-rag benchmark smoke --dataset scifact --branches A,B,D \
  --no-fixture-jev --skip-generation --config-path configs/openrouter-smoke.yaml

# Larger real Jev run: first set an explicit positive max_budget_usd in a copied config.
# The default of 0 blocks every paid call.
cp configs/paid.example.yaml configs/paid.yaml
uv run jev-rag benchmark full --dataset scifact --branches A,B,D --config-path configs/paid.yaml

# Generate CSV/JSONL-derived Turkish report and SVG
uv run jev-rag report results/scifact-a-b-d.jsonl
```

`benchmark full` uses every selected test query unless `--limit` is passed. A/B/D/O use the
same frozen top-20 candidates and exactly five context documents. E applies the dev-selected
threshold. P is pointwise Jev; R is the multi-signal evidence router; S is the permutation
ensemble; H is full-corpus hierarchical Jev; V is batch Jev plus citation verification.
Every generator prompt has the same 6,000-character maximum context budget.

## Tests

```bash
uv run pytest
uv run ruff check .
```

Tests cover retrieval metrics, hybrid retrieval, EM/F1 citation handling, paired bootstrap
reproducibility, Jev Noul/Choice response parsing, resolved-model logging, evidence routing,
hierarchical and permutation reranking, citation verification, sharding, concurrent budget
accounting, empty-generation retries, fallback behavior, and the zero-budget safety gate.

## Results and costs

Each run writes JSONL, a downloadable flat CSV, and a manifest containing commit,
dependency versions, dataset hashes, requested models, price date, seed, prompt versions,
cache mode, and concurrency. Reports separate retrieval, reranking, generation, and
end-to-end latency; Jev, generator, retry/fallback, and local compute are not conflated.
Generator model, reasoning effort, token cap, measured usage, finish reason, and resolved
OpenRouter model are recorded per row or manifest. Frozen-context replay prevents a fresh
retrieval draw from contaminating generator comparisons.

Mock/fixture output has `run_kind=fixture` and must never be cited as a real benchmark.
Local models have zero API price but non-zero measured runtime; the report calls this out.
SciFact qrels measure retrieval only. XQuAD EN/TR answer references measure end-to-end EM/F1,
and languages are reported separately.

## Data licenses

Downloaded data is ignored by Git and regenerated by scripts:

- SciFact claims/evidence annotations: CC BY 4.0; abstracts: ODC-By 1.0.
- XQuAD: CC BY-SA 4.0.

The project source is MIT. Upstream attributions are in [`NOTICE`](NOTICE).

## Current limitations

- Jev 1.13 is strongest in English; these Turkish results must not be generalized to other
  languages or domains without evaluation.
- Candidate order affects Jev scores. The measured ensemble reduces that sensitivity but is
  not the default because it triples cost for a small nDCG gain.
- Passage instructions are treated as untrusted data. The evidence router can label likely
  injection/contradiction, but this is not a complete security boundary.
- Automatic XQuAD F1, wrong-answer flags, and Jev citation decisions are proxies, not human
  factuality adjudication.
