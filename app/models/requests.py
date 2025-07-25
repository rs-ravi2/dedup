# app/models/requests.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class StoreMetadata(BaseModel):
    msisdn: str = Field(..., description="Customer phone number")
    created_on: Optional[str] = Field(None, description="Record creation timestamp (UTC)")
    id_type: str = Field(..., description="Type of identity document")
    id_number: str = Field(..., description="Identity document number")


class SearchMetadata(BaseModel):
    id_type: str = Field(..., description="Type of identity document")
    id_number: str = Field(..., description="Identity document number")
    created_on: str = Field(..., description="Record creation timestamp (UTC)")


class SearchRequest(BaseModel):
    threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0, description="Similarity threshold")
    limit: Optional[int] = Field(None, ge=1, description="Maximum results")


class PurgeRequest(BaseModel):
    transaction_id: str = Field(..., description="ID of the customer record to purge")