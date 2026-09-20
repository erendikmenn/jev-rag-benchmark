from pathlib import Path

from jev_rag_benchmark.io import read_jsonl
from jev_rag_benchmark.models import Document
from jev_rag_benchmark.retrieval import BM25Index, DenseIndex, HybridIndex


FIXTURES = Path(__file__).parent / "fixtures"


def test_bm25_retrieves_answer_document():
    docs = [Document(**row) for row in read_jsonl(FIXTURES / "documents.jsonl")]
    results = BM25Index(docs).retrieve("capital Türkiye", top_k=3)
    assert results[0].document.doc_id == "d1"
    assert [item.retrieval_rank for item in results] == [1, 2, 3]


class FakeEmbedder:
    model = "fake-embedding"
    last_cost_usd = 0.0
    last_input_tokens = 0
    resolved_model = model

    def embed(self, texts, *, input_type=None):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append(
                [
                    float("ankara" in lowered or "türkiye" in lowered),
                    float("paris" in lowered or "france" in lowered),
                ]
            )
        return vectors


def test_dense_and_hybrid_retrieve_semantic_match(tmp_path):
    docs = [
        Document("d1", "Ankara is the capital of Türkiye."),
        Document("d2", "Paris is the capital of France."),
    ]
    dense = DenseIndex(docs, FakeEmbedder(), cache_path=tmp_path / "embeddings.json")
    assert dense.retrieve("Türkiye Ankara", 1)[0].document.doc_id == "d1"

    hybrid = HybridIndex(BM25Index(docs), dense, source_k=2)
    selected = hybrid.retrieve("Türkiye Ankara", 2)
    assert selected[0].document.doc_id == "d1"
    assert [item.retrieval_rank for item in selected] == [1, 2]
