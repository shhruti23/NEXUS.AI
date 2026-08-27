from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class HybridSearchEngine:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        # Simple whitespace & lowercasing tokenization for fast CPU execution
        self.tokenized_corpus = [doc["content"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def search_bm25(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Performs sparse keyword matching using the Okapi BM25 algorithm."""
        if not self.bm25 or not self.documents:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        
        return [
            {
                "content": self.documents[i]["content"],
                "metadata": self.documents[i]["metadata"],
                "score": float(scores[i])
            }
            for i in top_indices if scores[i] > 0
        ]

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Fuses ranked lists using Reciprocal Rank Fusion (RRF).
        Formula: RRF_Score = sum(1 / (k + rank_i))
        """
        rrf_scores = {}
        doc_lookup = {}

        # Process dense vector hits
        for rank, doc in enumerate(dense_results):
            cid = doc["metadata"]["chunk_id"]
            doc_lookup[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))

        # Process sparse BM25 hits
        for rank, doc in enumerate(sparse_results):
            cid = doc["metadata"]["chunk_id"]
            if cid not in doc_lookup:
                doc_lookup[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))

        # Sort by highest reciprocal rank score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_lookup[cid] for cid, _ in sorted_chunks]