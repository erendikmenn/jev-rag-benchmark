# Upstream baseline reproduction

`app.py` is the selected notebook's Python application cell with formatting-only changes:

- Upstream repository: `run-llama/llama_index`
- Path: `docs/examples/usecases/fastapi_rag_ollama.ipynb`
- Commit: `f475afd8a9bbda84f252567e045d89d07b5701b3`
- License: MIT

It intentionally retains upstream's `llama3`, `nomic-embed-text`, `VectorStoreIndex`, and
FastAPI choices. The three tiny text documents only make its behavior quick to verify.

```bash
uv sync --extra llamaindex
ollama pull llama3
ollama pull nomic-embed-text
cd reproduction/upstream_baseline
uv run python smoke.py
```

This reproduction is separate from the controlled experiment so baseline changes cannot be
silently mistaken for unchanged upstream behavior.

## Verified run

Verified on 2026-09-19 on Apple Silicon with Ollama 0.34.2, LlamaIndex Core 0.14.24,
`llama3` (digest prefix `365c0bd3c000`) and `nomic-embed-text` (digest prefix
`0a109f422b47`). The unchanged app returned:

| Question | Answer | First source |
|---|---|---|
| What is the capital of Türkiye? | Ankara | `turkiye.txt` |
| Which city spans Europe and Asia? | Istanbul. | `turkiye.txt` |
| Why is Mars called the Red Planet? | Iron oxides give its surface a reddish appearance. | `planets.txt` |
