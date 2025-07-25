from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, Set


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
    api_key: str = "COcEJgOmhw0bwjhZdHwxDxWee7ZGGBRj"

    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: str = "image/jpeg,image/png,image/jpg"  # Store as string, convert to set

    @field_validator('allowed_image_types')
    @classmethod
    def parse_allowed_image_types(cls, v):
        if isinstance(v, str):
            return set(v.split(','))
        return v

    @property
    def allowed_image_types_set(self) -> Set[str]:
        """Get allowed image types as a set"""
        if isinstance(self.allowed_image_types, str):
            return set(self.allowed_image_types.split(','))
        return self.allowed_image_types

    class Config:
        env_file = ".env"

    class MinioConfig:
        # MinIO connection details
        minio_url: str = '172.24.4.37:9000'
        minio_username: str = 'datascience'
        minio_password: str = 'datascience'
        minio_bucket_name: str = 'ng-auto-cm-models'
        secure: bool = False
        model_object_path: str = "deduplication/models.zip"
        download_path: str = "/tmp"

settings = Settings()

#
# from pydantic_settings import BaseSettings
# from typing import Optional, Set
#
# class Settings(BaseSettings):
#     # API Configuration
#     api_title: str = "Customer Deduplication API"
#     api_version: str = "v1"
#     api_description: str = "API for customer deduplication using image similarity"
#
#     # Redis Configuration
#     redis_url: str = "redis://10.190.192.51:6379"
#     redis_password: Optional[str] = "Airtel#321"
#     redis_db: int = 0
#
#     # Vector Configuration
#     vector_dimension: int = 512
#     default_similarity_threshold: float = 0.8
#     max_search_results: int = 100
#
#     # Authentication
#     api_key: str = "COcEJgOmhw0bwjhZdHwxDxWee7ZGGBRj"
#
#     # File Upload
#     max_file_size: int = 10 * 1024 * 1024  # 10MB
#     allowed_image_types: Set[str] = {"image/jpeg", "image/png", "image/jpg"}
#
#     class Config:
#         env_file = ".env"
#
# class MinioConfig:
#     # MinIO connection details
#     minio_url:str =  '172.24.4.37:9000'
#     minio_username: str = 'datascience'
#     minio_password: str =  'datascience'
#     minio_bucket_name:str = 'ng-auto-cm-models'
#     secure: bool = False
#     model_object_path: str = "deduplication/models.zip"
#     download_path: str = "/tmp"
#
# settings = Settings()