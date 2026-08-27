import pytest
from app.services.embedding import VectorStoreService

def test_embedding_and_qdrant():
    service = VectorStoreService()
    
    sample_doc = [{
        "content": "Local enterprise search systems require air-gapped security and high indexing speed.",
        "metadata": {
            "source": "test_policy.txt",
            "page": 1,
            "chunk_id": "test_policy_p1_c0"
        }
    }]
    
    # Test insertion
    inserted = service.insert_documents(sample_doc)
    assert inserted == 1
    
    # Test vector retrieval
    hits = service.search_dense("air-gapped security", limit=1)
    assert len(hits) > 0
    assert "air-gapped" in hits[0]["content"]
    assert hits[0]["metadata"]["source"] == "test_policy.txt"