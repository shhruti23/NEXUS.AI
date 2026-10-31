from qdrant_client import QdrantClient

from app.services.embedding import VectorStoreService


class MockEmbeddings:

    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 768


def test_embedding_and_qdrant():

    client = QdrantClient(":memory:")

    embeddings = MockEmbeddings()

    service = VectorStoreService(
        client=client,
        embeddings=embeddings
    )

    sample_doc = [{
        "content": "Local enterprise search systems require air-gapped security and high indexing speed.",
        "metadata": {
            "source": "test_policy.txt",
            "page": 1,
            "chunk_id": "test_policy_p1_c0"
        }
    }]

    inserted = service.insert_documents(sample_doc)

    assert inserted == 1

    hits = service.search_dense(
        "air-gapped security",
        limit=1
    )

    assert len(hits) > 0
    assert "air-gapped" in hits[0]["content"]
    assert hits[0]["metadata"]["source"] == "test_policy.txt"

    client.close()
