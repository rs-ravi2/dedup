from pydantic import BaseModel, Field
from typing import List, Optional
from app.models.requests import CustomerMetadata

class BaseResponse(BaseModel):
    status: str = Field(..., description="Response status")

class StoreResponse(BaseResponse):
    customer_id: str = Field(..., description="Unique customer identifier")
    message: str = Field(..., description="Success message")

class SearchResult(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    similarity_score: float = Field(..., description="Similarity score")
    metadata: CustomerMetadata = Field(..., description="Customer metadata")

class SearchResponse(BaseResponse):
    total_matches: int = Field(..., description="Number of matching records found")
    results: List[SearchResult] = Field(..., description="List of matching records")

class PurgeResponse(BaseResponse):
    customer_id: str = Field(..., description="ID of the purged customer record")
    message: str = Field(..., description="Success message")

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[str] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail