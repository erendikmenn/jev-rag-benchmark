# Contributing

Use Python 3.11–3.13 and `uv`. Before opening a change:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Do not commit API keys, downloaded datasets, model weights, or outputs that could violate a
source dataset/model license. Mark every fixture/mock result as such. Changes to prompts,
prices, models, data splits, thresholds, or fallback policy must update the corresponding
version/config field so old results remain interpretable.

