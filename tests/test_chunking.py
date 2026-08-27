import pytest
from app.services.chunking import DocumentProcessor

def test_text_chunking():
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    raw_text = "FastAPI is a modern web framework. It is fast, scalable, and easy to learn." * 5
    chunks = processor.process_file("sample.txt", raw_text.encode("utf-8"))
    
    assert len(chunks) > 0
    assert chunks[0]["metadata"]["source"] == "sample.txt"
    assert "content" in chunks[0]