from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Optional
import json
from app.models.requests import StoreMetadata, SearchMetadata, SearchRequest, PurgeRequest
from app.models.responses import StoreResponse, SearchResponse, PurgeResponse
from app.services.dedup_service import dedup_service
from app.utils.auth import verify_token
from app.utils.exceptions import DedupException, create_http_exception
from app.config import settings

router = APIRouter(prefix="/v1/dedup/face", tags=["deduplication"])


@router.post("/store", response_model=StoreResponse)
async def store_record(
        image: UploadFile = File(..., description="Customer image file"),
        transaction_id: str = Form(..., description="Unique transaction identifier"),
        metadata: str = Form(..., description="JSON string containing customer metadata"),
        token: str = Depends(verify_token)
):
    """
    Store a new customer record with image and metadata.
    Generates embeddings and stores them in the vector database.
    """
    try:
        # Validate image content type
        if image.content_type not in settings.allowed_image_types:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": f"Unsupported image format: {image.content_type}",
                        "details": f"Allowed formats: {', '.join(settings.allowed_image_types)}"
                    }
                }
            )

        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
            store_metadata = StoreMetadata(**metadata_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "Invalid metadata format",
                        "details": str(e)
                    }
                }
            )

        # Read image data
        image_data = await image.read()

        # Store customer record
        await dedup_service.store_customer(
            transaction_id,
            image_data,
            store_metadata
        )

        return StoreResponse(
            status="success",
            transaction_id=transaction_id,
            message="Record inserted successfully"
        )

    except DedupException as e:
        raise create_http_exception(e)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                }
            }
        )


@router.post("/search", response_model=SearchResponse)
async def search_similar(
        image: UploadFile = File(..., description="Query image for similarity search"),
        transaction_id: str = Form(..., description="Unique transaction identifier"),
        metadata: str = Form(..., description="JSON string containing customer metadata"),
        threshold: Optional[float] = Form(default=0.6, description="Similarity threshold (0.0-1.0)"),
        limit: Optional[int] = Form(default=None, description="Maximum results"),
        token: str = Depends(verify_token)
):
    """
    Search for matching customer records based on image similarity.
    Uses vector embeddings and configurable similarity thresholds.
    """
    try:
        # Validate image content type
        if image.content_type not in settings.allowed_image_types:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": f"Unsupported image format: {image.content_type}",
                        "details": f"Allowed formats: {', '.join(settings.allowed_image_types)}"
                    }
                }
            )

        # Parse metadata
        try:
            metadata_dict = json.loads(metadata)
            search_metadata = SearchMetadata(**metadata_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "Invalid metadata format",
                        "details": str(e)
                    }
                }
            )

        # Validate search parameters
        if threshold is not None and (threshold < 0.0 or threshold > 1.0):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "Threshold must be between 0.0 and 1.0",
                        "details": None
                    }
                }
            )

        if limit is not None and limit < 1:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": "Limit must be greater than 0",
                        "details": None
                    }
                }
            )

        # Read image data
        image_data = await image.read()

        # Search for similar customers
        results = await dedup_service.search_similar_customers(
            image_data,
            threshold or 0.6,
            limit
        )

        # Convert results to match API specification
        api_results = []
        for result in results:
            api_results.append({
                "similarity_score": result.similarity_score,
                "metadata": result.metadata.dict()
            })

        return SearchResponse(
            status="success",
            transaction_id=transaction_id,
            total_matches=len(api_results),
            metadata=search_metadata,
            results=api_results
        )

    except DedupException as e:
        raise create_http_exception(e)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                }
            }
        )


@router.post("/purge", response_model=PurgeResponse)
async def purge_record(
        request: PurgeRequest,
        token: str = Depends(verify_token)
):
    """
    Delete a customer record from the system.
    Removes both vector embeddings and metadata.
    """
    try:
        # Purge customer record
        await dedup_service.purge_customer(request.transaction_id)

        return PurgeResponse(
            status="success",
            transaction_id=request.transaction_id,
            message="Record purged successfully"
        )

    except DedupException as e:
        raise create_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error",
                    "details": str(e)
                }
            }
        )


@router.get("/health")
async def health_check():
    """Check service health status"""
    try:
        health_status = await dedup_service.health_check()
        status_code = 200 if health_status["overall"] else 503

        return {
            "status": "healthy" if health_status["overall"] else "unhealthy",
            "services": health_status
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }