# Jev permutation ensemble comparison

The comparison uses the same 200 Turkish XQuAD questions and the same hybrid
full-depth top-20 candidate pool. Generation is disabled so that the measurement
isolates reranking.

| Method | Recall@5 | nDCG@10 | MRR@10 | Rerank p50 | Rerank cost |
|---|---:|---:|---:|---:|---:|
| Jev batch (1 order) | 99.00% | 97.893% | 97.500% | 529.2 ms | $0.078964 |
| Jev permutation ensemble (3 orders) | 99.00% | 98.131% | 97.833% | 586.6 ms | $0.236928 |

The ensemble preserved the same two misses and improved ordering quality by
0.238 nDCG points and 0.333 MRR points. Because the three requests run in
parallel, median reranking latency rose by only 10.9%, but API cost tripled.
This is useful as an optional stability mode, not as the default production
path: the single-order Jev batch remains the stronger quality/cost choice.
