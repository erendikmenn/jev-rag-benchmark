# Jev evidence router comparison

The comparison uses the same 200 Turkish XQuAD questions and the same hybrid
full-depth top-20 candidate pool. Generation is disabled.

| Method | Recall@5 | nDCG@10 | MRR@10 | Rerank p50 | Rerank cost | Fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Jev batch relevance | 99.00% | 97.893% | 97.500% | 529.2 ms | $0.078964 | 0 |
| Jev multi-signal evidence router | 93.00% | 92.381% | 92.167% | 1021.8 ms | $0.155189 | 10 |

The evidence router was too selective for this extractive-QA benchmark. It
lost 12 gold passages relative to the batch relevance scorer, nearly doubled
median reranking latency, and nearly doubled cost. The richer route labels are
still useful for untrusted or contradictory corpora, but this configuration
must not replace the default XQuAD reranker without a separately calibrated
gate.
