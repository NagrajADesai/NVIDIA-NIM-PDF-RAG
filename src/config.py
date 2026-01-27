import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class AppConfig:
    """Application configuration settings."""
    APP_TITLE: str = "NIMbleRAG: Advanced RAG Agent"
    APP_ICON: str = "🧠"
    LAYOUT: str = "wide"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    VECTOR_DB_DIR: str = "vector_dbs"

@dataclass
class ModelConfig:
    """Model configuration settings."""
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY")
    LLM_MODEL: str = "meta/llama-3.1-8b-instruct"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
