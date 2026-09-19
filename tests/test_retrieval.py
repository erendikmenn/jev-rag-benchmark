from pathlib import Path

from jev_rag_benchmark.io import read_jsonl
from jev_rag_benchmark.models import Document
from jev_rag_benchmark.retrieval import BM25Index


FIXTURES = Path(__file__).parent / "fixtures"


def test_bm25_retrieves_answer_document():
    docs = [Document(**row) for row in read_jsonl(FIXTURES / "documents.jsonl")]
    results = BM25Index(docs).retrieve("capital Türkiye", top_k=3)
    assert results[0].document.doc_id == "d1"
    assert [item.retrieval_rank for item in results] == [1, 2, 3]

