from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Configuration
    api_title: str = "Telecom Customer Deduplication API"
    api_version: str = "v1"
    api_description: str = "API for telecom customer deduplication using image similarity"

    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_db: int = 0

    # Vector Configuration
    vector_dimension: int = 512  # Embedding dimension
    default_similarity_threshold: float = 0.8
    max_search_results: int = 100

    # Authentication
    api_key: str = "your-api-key-here"  # In production, use proper secret management

    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: set = {"image/jpeg", "image/png", "image/jpg"}

    class Config:
        env_file = ".env"


settings = Settings()