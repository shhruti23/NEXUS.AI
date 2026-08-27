from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
from app.api.v1.schemas import QueryRequest, UploadResponse
from app.services.chunking import DocumentProcessor
from app.services.embedding import VectorStoreService
from app.services.reranker import DocumentReranker
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()

processor = DocumentProcessor()
vector_store = VectorStoreService()
reranker = DocumentReranker()
pipeline = RAGPipeline(vector_store, reranker)

CORPUS_REGISTRY: List[dict] = []
DOCUMENT_NAMES: List[str] = []


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    total_chunks = 0
    for file in files:
        content = await file.read()
        try:
            chunks = processor.process_file(file.filename, content)
            if chunks:
                vector_store.insert_documents(chunks)
                CORPUS_REGISTRY.extend(chunks)
                if file.filename not in DOCUMENT_NAMES:
                    DOCUMENT_NAMES.append(file.filename)
                total_chunks += len(chunks)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process {file.filename}: {str(e)}")

    return UploadResponse(
        status="success",
        processed_files=len(files),
        total_chunks=total_chunks
    )


@router.get("/documents")
async def list_documents():
    return {
        "documents": DOCUMENT_NAMES,
        "total_chunks": len(CORPUS_REGISTRY)
    }


@router.delete("/clear")
async def clear_collection():
    global CORPUS_REGISTRY, DOCUMENT_NAMES
    vector_store.clear_collection()
    CORPUS_REGISTRY = []
    DOCUMENT_NAMES = []
    return {"status": "success", "message": "Knowledge base and Qdrant collection cleared."}


@router.post("/query")
async def query_pipeline(request: QueryRequest):
    return StreamingResponse(
        pipeline.stream_query(request.query, CORPUS_REGISTRY, mode=request.mode),
        media_type="text/event-stream"
    )