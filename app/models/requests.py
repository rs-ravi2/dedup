# app/models/requests.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class CustomerMetadata(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    msisdn: str = Field(..., description="Customer phone number")
    created_on: str = Field(..., description="Record creation timestamp")
    id_type: str = Field(..., description="Type of identity document")
    id_number: str = Field(..., description="Identity document number")


class SearchRequest(BaseModel):
    threshold: Optional[float] = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum results")


class PurgeRequest(BaseModel):
    customer_id: str = Field(..., description="ID of the customer record to purge")
