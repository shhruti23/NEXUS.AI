import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_txt_file():
    file_content = b"This is an enterprise policy document regarding data storage."
    response = client.post(
        "/api/v1/upload",
        files={"files": ("policy.txt", file_content, "text/plain")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["processed_files"] == 1