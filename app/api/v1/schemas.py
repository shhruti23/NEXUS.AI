from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="The query to ask the document repository.")
    mode: str = Field("hybrid", description="'hybrid' (Dense + BM25) or 'dense' (Dense only)")

class UploadResponse(BaseModel):
    status: str
    processed_files: int
    total_chunks: int