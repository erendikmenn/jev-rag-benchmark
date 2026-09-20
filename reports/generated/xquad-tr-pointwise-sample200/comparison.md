# Jev batch vs pointwise comparison

The comparison uses the same 200 Turkish XQuAD questions and the same hybrid
full-depth top-20 candidate pool. Generation is disabled.

| Method | Recall@5 | nDCG@10 | MRR@10 | Rerank p50 | Rerank cost |
|---|---:|---:|---:|---:|---:|
| Jev batch | 99.00% | 97.893% | 97.500% | 529.2 ms | $0.078964 |
| Jev pointwise | 99.00% | 97.008% | 96.333% | 956.8 ms | $0.123589 |

Pointwise scoring recovered no additional gold passages. It reduced ordering
quality, increased median reranking latency by 80.8%, and increased API cost by
56.5%. Batch scoring is therefore the default for this corpus.
