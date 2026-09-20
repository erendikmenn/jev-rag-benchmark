# Jev citation verification comparison

This 100-query ablation compares the existing full Gemini replay with a fresh
run of the standard Jev batch reranker plus claim-level Jev citation
verification. Both use the same sampled XQuAD questions and the same hybrid
full-depth top-20 retrieval setup. Jev API scoring is not bit-deterministic, so
the fresh rerank produced a slightly different nDCG while preserving Recall@5.

| Metric | Jev + Gemini | + Jev citation verification |
|---|---:|---:|
| Recall@5 | 99.00% | 99.00% |
| nDCG@10 | 97.155% | 97.693% |
| Mean answer F1 | 26.591% | 27.497% |
| Successful answers | 16/100 | 17/100 |
| Syntactically valid citations | 95.00% | 95.00% |
| Semantically supported cited claims | not measured | 100.00% |
| Regenerations | 0 | 1 |
| p50 end-to-end latency | 4216.6 ms | 5007.8 ms |
| Total online cost | $0.252133 | $0.261499 |

The verifier added $0.003106 of direct Jev verification cost and triggered one
corrective generation. All final cited claims were judged supported. The total
sample cost rose by 3.7% and median end-to-end latency by 18.8%. This makes
verification a useful high-assurance mode, while leaving it optional for the
lowest-latency path.
