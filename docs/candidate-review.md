# Baseline candidate review (2026-09-19)

The review was intentionally capped at three candidates. Repository activity, exact paths,
default branches, and HEAD commits were checked on the review date.

| Candidate | Exact source | License | Setup / macOS | Retrieval intervention | Decision |
|---|---|---|---|---|---|
| LlamaIndex FastAPI + Ollama RAG | `run-llama/llama_index/docs/examples/usecases/fastapi_rag_ollama.ipynb` at `f475afd8a9bbda84f252567e045d89d07b5701b3` | MIT | Local Ollama; no paid key; upstream defaults download `llama3` and `nomic-embed-text` | Easy: query engine can be separated at its retriever/postprocessor boundary | Selected |
| LlamaIndex RAG Workflow with Reranking | `run-llama/llama_index/docs/examples/workflow/rag.ipynb` at the same commit | MIT | Requires OpenAI key and downloads an arXiv PDF | Easy, but the example already contains an LLM reranker, confounding branch A | Rejected |
| Haystack FastEmbed RAG cookbook | `deepset-ai/haystack-cookbook/notebooks/rag_fastembed.ipynb` at `35212da9a2dfe6374b02f11c5f1a541265b09c1d` | No root license found; an open upstream issue asks for clarification | Local FastEmbed; notebook-oriented | Easy | Rejected: code was not copied |

The selected repository was active and its documented RAG API exposes retrievers and node
postprocessors. Its full default local model stack is heavy, so the controlled benchmark
records one baseline change: deterministic BM25 candidates replace embedding retrieval.
The answer generator remains local Ollama and is identical across branches. This avoids
claiming that a changed local model is an “unchanged” upstream execution.

