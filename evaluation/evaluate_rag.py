import json
import os
import time
from app.services.embedding import VectorStoreService
from app.services.reranker import DocumentReranker
from app.services.hybrid_search import HybridSearchEngine

def run_benchmark():
    dataset_path = os.path.join(os.path.dirname(__file__), "test_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        test_cases = json.load(f)

    vector_store = VectorStoreService()
    reranker = DocumentReranker()

    # Synthetic corpus simulation
    mock_corpus = [
        {
            "content": "Enterprise offline authentication uses AES-256 token encryption standards with local key rings.",
            "metadata": {"source": "enterprise_security_protocol.txt", "page": 1, "chunk_id": "sec_01"}
        },
        {
            "content": "Data persistence in air-gapped environments relies on Qdrant local write-ahead logs.",
            "metadata": {"source": "storage_architecture.md", "page": 1, "chunk_id": "stor_01"}
        },
        {
            "content": "General employee travel reimbursement policies and guidelines.",
            "metadata": {"source": "hr_handbook.pdf", "page": 3, "chunk_id": "hr_03"}
        }
    ]

    vector_store.insert_documents(mock_corpus)

    print("\n" + "=" * 50)
    print(" 🚀 NEXUS.AI Retrieval Benchmark Runner")
    print("=" * 50)

    for mode in ["Dense Only", "Hybrid RRF + Neural Re-ranking"]:
        hits_at_1 = 0
        hits_at_3 = 0
        mrr_sum = 0.0
        total_latency = 0.0

        for case in test_cases:
            t0 = time.perf_counter()
            dense_hits = vector_store.search_dense(case["query"], limit=5)

            if mode == "Dense Only":
                final_results = dense_hits[:3]
            else:
                hybrid_engine = HybridSearchEngine(mock_corpus)
                sparse_hits = hybrid_engine.search_bm25(case["query"], limit=5)
                fused = HybridSearchEngine.reciprocal_rank_fusion(dense_hits, sparse_hits)
                final_results = reranker.rerank(case["query"], fused, top_k=3)

            latency = (time.perf_counter() - t0) * 1000
            total_latency += latency

            retrieved_sources = [d["metadata"]["source"] for d in final_results]

            # Metric Calculations
            if retrieved_sources and retrieved_sources[0] == case["expected_source"]:
                hits_at_1 += 1
            if case["expected_source"] in retrieved_sources[:3]:
                hits_at_3 += 1
                rank = retrieved_sources.index(case["expected_source"]) + 1
                mrr_sum += 1.0 / rank

        n = len(test_cases)
        print(f"\nMode: {mode}")
        print(f" • Hit Rate @ 1 : {(hits_at_1 / n) * 100:.1f}%")
        print(f" • Hit Rate @ 3 : {(hits_at_3 / n) * 100:.1f}%")
        print(f" • MRR Score    : {(mrr_sum / n):.3f}")
        print(f" • Avg Latency  : {total_latency / n:.2f} ms")

    print("\n" + "=" * 50)

if __name__ == "__main__":
    run_benchmark()