# Full-corpus hierarchical Jev comparison

This 100-query ablation gives Jev access to all 240 XQuAD Turkish passages.
Jev first scores eight 30-passage shards, keeps five candidates per shard, and
then scores the 40 finalists. The baseline is BM25 top-5 on the same queries.

| Method | Recall@5 | nDCG@10 | p50 end-to-end | Cost/query |
|---|---:|---:|---:|---:|
| BM25 top-5 | 97.00% | 91.56% | 1.7 ms | $0.000000 |
| Hierarchical Jev over all 240 passages | 76.00% | 61.21% | 1183.7 ms | $0.005061 |

The hierarchy loses relevant passages at the shard-pruning stage and is both
less accurate and much more expensive than first-stage retrieval. Jev should
therefore remain a narrow decision layer over a strong retriever, not replace
the retriever with full-corpus tournament scoring.
