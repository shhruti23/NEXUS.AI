import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings


class VectorStoreService:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.EMBEDDING_MODEL
        )
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == settings.COLLECTION_NAME for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=768,
                    distance=qmodels.Distance.COSINE
                )
            )

    def clear_collection(self):
        """Drops and recreates the collection to reset state."""
        self.client.delete_collection(collection_name=settings.COLLECTION_NAME)
        self._ensure_collection()

    def insert_documents(self, documents: List[Dict[str, Any]]) -> int:
        if not documents:
            return 0
            
        texts = [doc["content"] for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={
                    "content": documents[i]["content"],
                    "source": documents[i]["metadata"]["source"],
                    "page": documents[i]["metadata"]["page"],
                    "chunk_id": documents[i]["metadata"]["chunk_id"]
                }
            )
            for i in range(len(documents))
        ]
        
        self.client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points
        )
        return len(points)

    def search_dense(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query_vector = self.embeddings.embed_query(query)
        
        if hasattr(self.client, "query_points"):
            results = self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=query_vector,
                limit=limit
            ).points
        else:
            results = self.client.search(
                collection_name=settings.COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit
            )

        return [
            {
                "content": hit.payload["content"],
                "metadata": {
                    "source": hit.payload["source"],
                    "page": hit.payload["page"],
                    "chunk_id": hit.payload["chunk_id"]
                },
                "score": getattr(hit, "score", 0.0)
            }
            for hit in results
        ]