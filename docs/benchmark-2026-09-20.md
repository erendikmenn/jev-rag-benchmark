# XQuAD-TR benchmark — 2026-09-20

This report records the locked OpenRouter experiments. It separates candidate
retrieval, reranking, answer generation, and citation verification so that a
gain in one stage is not attributed to another.

## Scope and controls

- Dataset: Google/DeepMind XQuAD, Turkish test split.
- Evaluation set: 1,044 unique questions after normalized-text deduplication.
- Corpus: 240 Turkish passages.
- First stage: BM25 + OpenRouter `baai/bge-m3`, weighted reciprocal-rank fusion.
- Fusion depth: all 240 passages; 20 candidates are exposed to rerankers.
- Reranker output: five passages with a 6,000-character prompt budget.
- Jev request: OpenRouter Decisions API, requested `typesafe/jev-1.13`, resolved
  `typesafe/jev-1.13-20260917`.
- Random seed: `20260919`.
- “Successful answer”: citation markers removed, then token F1 >= 0.5 against
  an XQuAD reference answer.

The generator comparison replays the exact frozen Jev context IDs. It therefore
measures the generator rather than a fresh retrieval/reranking draw.

## Retrieval ceiling

| Candidate source | k | Gold passage available | Recall |
|---|---:|---:|---:|
| BM25 | 5 | 979/1,044 | 93.774% |
| BM25 | 20 | 1,009/1,044 | 96.648% |
| BM25 | 50 | 1,020/1,044 | 97.701% |
| BM25 | 100 | 1,028/1,044 | 98.467% |
| BM25 | 240 | 1,044/1,044 | 100.000% |
| Hybrid BM25 + BGE-M3 | 5 | 1,019/1,044 | 97.605% |
| Hybrid BM25 + BGE-M3 | 20 | 1,039/1,044 | 99.521% |
| Hybrid BM25 + BGE-M3 | 50 | 1,044/1,044 | 100.000% |

This is the first important result: reranking cannot recover a gold passage
that was never placed in its candidate pool. Full-depth hybrid fusion raises
the top-20 ceiling from 96.648% to 99.521% before Jev is called.

## Full reranker comparison

Generation is disabled in this table. Every method receives the same hybrid
top-20 candidates.

| Method | Recall@5 | nDCG@10 | MRR@10 | Rerank p50 | Total rerank cost |
|---|---:|---:|---:|---:|---:|
| Hybrid order, no reranker | 97.605% | 94.314% | 93.194% | 0.0 ms | $0.000000 |
| Jev batch Noul | 99.425% | 98.097% | 97.637% | 532.0 ms | $0.410866 |
| Cohere Rerank 3.5 | 99.425% | 98.626% | 98.348% | 466.1 ms | $1.044000 |

Jev and Cohere each move the gold passage into top-5 for 1,038/1,044
questions. Jev costs 60.6% less than Cohere while giving up 0.529 nDCG points
and 65.9 ms at the median. Jev adds 19 top-5 recoveries over the raw hybrid
order, with zero fallbacks.

Raw rows and manifests are in
[`results/xquad-tr-hybrid-fullfusion20-a-d-o.jsonl`](../results/xquad-tr-hybrid-fullfusion20-a-d-o.jsonl),
and the generated report is in
[`reports/generated/xquad-tr-hybrid-fullfusion20-a-d-o/`](../reports/generated/xquad-tr-hybrid-fullfusion20-a-d-o/).

## Frozen-context generator comparison

| Metric | Gemini 3.8 Flash | DeepSeek V4.1 Flash |
|---|---:|---:|
| Resolved model | `google/gemini-3.8-flash` | `deepseek/deepseek-v4.1-flash` |
| Questions | 1,044 | 1,044 |
| Mean answer F1 | 27.274% | 29.191% |
| Exact match | 0.096% | 0.862% |
| Successful answers | 163 (15.61%) | 194 (18.58%) |
| Valid citation IDs | 93.87% | 95.69% |
| Abstention | 6.13% | 4.21% |
| Wrong-answer flag | 78.26% | 77.20% |
| Generation p50 / p95 | 3.128 s / 13.114 s | 7.641 s / 46.537 s |
| Generation cost | $2.457358 | $0.659245 |
| Jev + generation cost | $2.868223 | $1.070110 |

On paired questions, DeepSeek improves mean F1 by 1.91 points (95% bootstrap
CI +1.03 to +2.86) and success rate by 2.97 points (95% CI +0.96 to +5.08).
It wins/ties/loses on per-question F1 in 404/362/278 cases. Its generation
cost is 73.2% lower, but observed median generation latency is 2.44x Gemini's
and its p95 is 3.55x Gemini's.

The practical choice is therefore workload-dependent:

- Use DeepSeek V4.1 Flash when quality and price dominate and long-tail latency
  is acceptable.
- Use Gemini 3.8 Flash when interactive latency is more important than the
  measured 2.97-point success-rate gap.

See [`gemini-vs-deepseek.md`](../reports/generated/gemini-vs-deepseek.md) for
the paired report.

## Jev ablations

| Experiment | Sample | Outcome | Decision |
|---|---:|---|---|
| Pointwise scoring | 200 | Same 99.0% Recall@5; −0.885 nDCG points; +80.8% p50 rerank latency; +56.5% cost | Keep batch as default |
| Three-permutation ensemble | 200 | Same 99.0% Recall@5; +0.238 nDCG points; +10.9% p50 rerank latency; 3x cost | Optional stability mode |
| Multi-signal evidence router | 200 | Recall@5 fell from 99% to 93%; about 2x latency and cost | Security/contradiction mode only; calibrate separately |
| Full-corpus hierarchical Jev | 100 | Recall@5 76% versus BM25's 97%; $0.005061/query | Do not replace the retriever with Jev |
| Citation verification | 100 | Recall@5 stayed 99%; all final cited claims verified; one regeneration | Optional high-assurance mode |

The order-stability audit used 50 questions and five candidate permutations.
Mean Spearman rank correlation was 0.262, mean top-5 Jaccard was 0.288, and
gold top-5 membership changed for 3/50 questions. The ensemble reduces this
sensitivity, but its marginal quality gain does not justify the 3x default
cost on XQuAD.

The development-only threshold calibration selected 0.50 at 95% recall
(77.78% precision, F1 85.53%). Thresholded evidence filtering remains a
separate mode; the best full test result above uses unthresholded batch ranking.

## Citation verifier

The verifier decomposes the generated answer into cited claims. For each
claim/source pair, Jev returns a typed `supports`, `contradicts`, or
`says_nothing` decision. Fabricated source IDs and uncited answers are rejected
without asking the model. A failed answer can be regenerated once with the
structured failure list and then reverified.

On the 100-question sample, direct verification cost was $0.003106. Total
online cost rose 3.7% and median end-to-end latency rose 18.8%. The reported
100% semantic support applies only to the final cited claims in this sample;
it is not a general factual-correctness guarantee.

## Interpretation limits

- Retrieval Recall/nDCG and answer F1 measure different stages. A 99.425%
  Recall@5 does not imply 99.425% correct generated answers.
- XQuAD references are short extractive answers. The generator was instructed
  to write up to two sourced sentences, so exact match is intentionally harsh.
- The wrong-answer flag means non-abstaining F1 < 0.5; it is an automatic proxy,
  not human adjudication.
- Latency is the observed OpenRouter route latency during this run. It is not a
  universal provider TPS guarantee.
- Model prices, aliases, and provider routing can change. Every manifest records
  requested/resolved model IDs, price date, prompt/config, and dataset hashes.
