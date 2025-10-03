"""
Configuration settings for the RAG application
"""

from pydantic_settings import BaseSettings, SettingsConfigDict  #
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "RAG PDF Chat API"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Storage
    STORAGE_PATH: Path = Path("./data")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_PAGES_PER_PDF: int = 70
    ALLOWED_EXTENSIONS: list[str] = [".pdf"]

    # Vector Store
    VECTOR_STORE_TYPE: str = "faiss"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # Embeddings
    DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    BASE_URL_PROVIDER: str = "https://router.huggingface.co/v1"
    DEFAULT_LLM_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct:together"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 800

    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None

    # Retrieval
    RETRIEVAL_K: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        # Asegura la búsqueda de variables de entorno del sistema (SHELL)
        extra="ignore",
    )


settings = Settings()
