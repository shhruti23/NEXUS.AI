from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "NEXUS.AI | Enterprise Offline RAG"
    
    # Qdrant configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "enterprise_docs"
    
    # Ollama local inference settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma2:2b"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    
    # Cross-encoder re-ranking model
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Pipeline hyperparameters
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()