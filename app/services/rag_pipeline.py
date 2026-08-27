import json
import time
import httpx
from typing import AsyncGenerator, List, Dict, Any
from app.core.config import settings
from app.services.hybrid_search import HybridSearchEngine


class RAGPipeline:
    def __init__(self, vector_store, reranker):
        self.vector_store = vector_store
        self.reranker = reranker

    async def stream_query(
        self, query: str, raw_corpus: List[Dict[str, Any]], mode: str = "hybrid"
    ) -> AsyncGenerator[str, None]:
        start_retrieval = time.perf_counter()

        # 1. Dense retrieval
        dense_hits = self.vector_store.search_dense(query, limit=settings.TOP_K_RETRIEVAL)

        if mode == "dense":
            fused_candidates = dense_hits
        else:
            # 2. Sparse retrieval via BM25
            hybrid_engine = HybridSearchEngine(raw_corpus)
            sparse_hits = hybrid_engine.search_bm25(query, limit=settings.TOP_K_RETRIEVAL)

            # 3. Reciprocal Rank Fusion
            fused_candidates = HybridSearchEngine.reciprocal_rank_fusion(dense_hits, sparse_hits)

        # 4. Neural Cross-Encoder Re-ranking
        ranked_docs = self.reranker.rerank(query, fused_candidates, top_k=settings.TOP_K_RERANK)
        retrieval_latency_ms = round((time.perf_counter() - start_retrieval) * 1000, 2)

        if not ranked_docs:
            yield f"event: metadata\ndata: {json.dumps({'sources': [], 'telemetry': {'latency_ms': retrieval_latency_ms, 'mode': mode}})}\n\n"
            yield f"event: token\ndata: {json.dumps({'token': 'No matching context found. Ingest your documents to get started.'})}\n\n"
            return

        # Prepare source citations & telemetry
        sources = [
            {
                "source": doc["metadata"]["source"],
                "page": doc["metadata"]["page"],
                "chunk_id": doc["metadata"]["chunk_id"],
                "score": round(doc.get("rerank_score", doc.get("score", 0.0)), 4),
                "content": doc["content"]
            }
            for doc in ranked_docs
        ]
        
        telemetry = {
            "latency_ms": retrieval_latency_ms,
            "mode": mode,
            "top_score": sources[0]["score"] if sources else 0.0
        }

        yield f"event: metadata\ndata: {json.dumps({'sources': sources, 'telemetry': telemetry})}\n\n"

        # 5. Build prompt
        context = "\n\n".join([
            f"[Source: {d['metadata']['source']} (Page {d['metadata']['page']})]\n{d['content']}"
            for d in ranked_docs
        ])

        prompt = f"""You are a precise enterprise search assistant. Answer the user prompt based strictly on the provided context below. If the context does not contain the answer, state that you cannot find the answer in the provided documents.

Context:
{context}

Question: {query}
Answer:"""

        # 6. Stream tokens from Ollama
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True
                }
            ) as response:
                async for chunk in response.aiter_lines():
                    if chunk:
                        payload = json.loads(chunk)
                        token = payload.get("response", "")
                        if token:
                            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                        if payload.get("done", False):
                            break