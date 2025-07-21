from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Optional
import json
from app.models.requests import CustomerMetadata, SearchRequest, PurgeRequest
from app.models.responses import StoreResponse, SearchResponse, PurgeResponse
from app.services.dedup_service import dedup_service
from app.utils.auth import verify_token
from app.utils.exceptions import DedupException, create_http_exception
from app.config import settings

router = APIRouter(prefix="/v1/dedup/face", tags=["deduplication"])


@router.post("/store", response_model=StoreResponse)
async def store_record(
        image: UploadFile = File(..., description="Customer image file"),
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
            customer_metadata = CustomerMetadata(**metadata_dict)
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
            customer_metadata.customer_id,
            image_data,
            customer_metadata
        )

        return StoreResponse(
            status="success",
            customer_id=customer_metadata.customer_id,
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
        threshold: Optional[float] = Form(default=settings.default_similarity_threshold,
                                          description="Similarity threshold (0.0-1.0)"),
        limit: Optional[int] = Form(default=10, description="Maximum results"),
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

        if limit is not None and (limit < 1 or limit > settings.max_search_results):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "code": "INVALID_REQUEST",
                        "message": f"Limit must be between 1 and {settings.max_search_results}",
                        "details": None
                    }
                }
            )

        # Read image data
        image_data = await image.read()

        # Search for similar customers
        results = await dedup_service.search_similar_customers(
            image_data,
            threshold or settings.default_similarity_threshold,
            limit or 10
        )

        return SearchResponse(
            status="success",
            total_matches=len(results),
            results=results
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
        await dedup_service.purge_customer(request.customer_id)

        return PurgeResponse(
            status="success",
            customer_id=request.customer_id,
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