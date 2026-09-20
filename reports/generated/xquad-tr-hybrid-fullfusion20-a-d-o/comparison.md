# Full XQuAD-TR reranker comparison

All 1,044 unique Turkish XQuAD questions use the same hybrid BM25 + BGE-M3
candidate pool. Dense/BM25 fusion considers all 240 corpus passages and exposes
the best 20 candidates to each reranker. Generation is disabled.

| Method | Recall@5 | nDCG@10 | MRR@10 | Rerank p50 | Total rerank cost |
|---|---:|---:|---:|---:|---:|
| Hybrid retrieval order | 97.605% | 94.314% | 93.194% | 0.0 ms | $0.000000 |
| Jev 1.13 batch | 99.425% | 98.097% | 97.637% | 532.0 ms | $0.410866 |
| Cohere Rerank 3.5 | 99.425% | 98.626% | 98.348% | 466.1 ms | $1.044000 |

Jev and Cohere recover the same 1,038/1,044 gold passages into top-5. Jev is
slightly weaker on exact ordering (−0.529 nDCG points) and 65.9 ms slower at
the median, but costs 60.6% less. Relative to the raw hybrid order, Jev adds
1.820 Recall points and 3.783 nDCG points with zero fallbacks.
