from pydantic_settings import BaseSettings
from typing import Optional, Set

class Settings(BaseSettings):
    use_local_redis: bool = False

    # Local Redis (e.g., Docker-based)
    local_redis_host: str = "localhost"
    local_redis_port: int = 6379
    local_redis_db: int = 0
    local_redis_password: Optional[str] = None

    # Cloud Redis
    cloud_redis_host: str = "10.190.192.51"
    cloud_redis_port: int = 6379
    cloud_redis_db: int = 0
    cloud_redis_password: str = "Airtel#321"

    # Vector Configuration
    vector_dimension: int = 512
    default_similarity_threshold: float = 0.6
    max_search_results: int = 100

    # API Metadata
    api_title: str = "Customer Deduplication API"
    api_version: str = "v1"
    api_description: str = "API for customer deduplication using image similarity"

    # Authentication
    api_key: str = "COcEJgOmhw0bwjhZdHwxDxWee7ZGGBRj"

    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: Set[str] = {"image/jpeg", "image/png", "image/jpg"}

    class Config:
        env_file = ".env"


settings = Settings()