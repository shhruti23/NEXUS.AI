from app.services.hybrid_search import HybridSearchEngine
from app.services.reranker import DocumentReranker

def test_rrf_and_reranker():
    docs = [
        {"content": "Python is a versatile programming language used in AI.", "metadata": {"chunk_id": "c1", "source": "f1", "page": 1}},
        {"content": "Ollama runs large language models locally on consumer CPUs.", "metadata": {"chunk_id": "c2", "source": "f2", "page": 1}},
        {"content": "FastAPI is designed for building fast APIs with Python.", "metadata": {"chunk_id": "c3", "source": "f3", "page": 1}}
    ]

    # Test BM25
    engine = HybridSearchEngine(docs)
    bm25_hits = engine.search_bm25("local Ollama models", limit=2)
    assert len(bm25_hits) > 0
    assert bm25_hits[0]["metadata"]["chunk_id"] == "c2"

    # Test RRF Fusion
    fused = HybridSearchEngine.reciprocal_rank_fusion(docs[:2], bm25_hits)
    assert len(fused) > 0

    # Test Re-ranker
    reranker = DocumentReranker()
    top_docs = reranker.rerank("running LLMs on laptops", fused, top_k=1)
    assert len(top_docs) == 1
    assert "Ollama" in top_docs[0]["content"]