from fastapi.testclient import TestClient

from app.main import app
from app.api.v1 import endpoints

client = TestClient(app)


def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_upload_txt_file(monkeypatch):

    monkeypatch.setattr(
        endpoints.vector_store,
        "insert_documents",
        lambda documents: len(documents)
    )

    file_content = (
        b"This is an enterprise policy document regarding data storage."
    )

    response = client.post(
        "/api/v1/upload",
        files={
            "files": (
                "policy.txt",
                file_content,
                "text/plain"
            )
        }
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["processed_files"] == 1
