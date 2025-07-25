# app/models/responses.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.models.requests import StoreMetadata, SearchMetadata


class BaseResponse(BaseModel):
    status: str = Field(..., description="Response status")


class StoreResponse(BaseResponse):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    message: str = Field(..., description="Success message")


class SearchResult(BaseModel):
    similarity_score: float = Field(..., description="Similarity score")
    metadata: StoreMetadata = Field(..., description="Customer metadata")


class SearchResponse(BaseResponse):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    total_matches: int = Field(..., description="Number of matching records found")
    metadata: SearchMetadata = Field(..., description="Query customer metadata")
    results: List[SearchResult] = Field(..., description="List of matching records")


class PurgeResponse(BaseResponse):
    transaction_id: str = Field(..., description="ID of the purged customer record")
    message: str = Field(..., description="Success message")


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail