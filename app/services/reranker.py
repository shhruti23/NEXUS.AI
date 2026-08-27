from sentence_transformers import CrossEncoder
from app.core.config import settings
from typing import List, Dict, Any

class DocumentReranker:
    def __init__(self):
        # ms-marco-MiniLM-L-6-v2 is ultra-lightweight and runs fast on CPU (<60ms)
        self.model = CrossEncoder(settings.RERANKER_MODEL)

    def rerank(self, query: str, candidate_docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Re-scores candidate chunks against the exact query using a cross-attention model."""
        if not candidate_docs:
            return []
        
        pairs = [[query, doc["content"]] for doc in candidate_docs]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            candidate_docs[i]["rerank_score"] = float(score)

        # Sort by neural cross-encoder score descending
        reranked_docs = sorted(candidate_docs, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs[:top_k]