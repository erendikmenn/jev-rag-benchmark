# Short screen demo (about 90 seconds)

1. Run `uv run jev-rag doctor`; point out that secret values are never printed.
2. Run `uv run jev-rag data prepare` and show the normalized file names and hashes in a run manifest.
3. Run `uv run jev-rag benchmark smoke --dataset scifact --branches A,B,D --fixture-jev --skip-generation`.
4. Open the JSONL and show that A/B/D have exactly the same `candidate_ids`.
5. Run `uv run jev-rag report results/scifact-a-b-d.jsonl` and open the Turkish report plus SVG.
6. Explain that `run_kind=fixture` prevents mock numbers from being mistaken for Jev results.
7. With an explicit budget and credentials, rerun without `--fixture-jev` and with a positive `run.max_budget_usd`.

