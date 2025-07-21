from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from typing import Optional, List
import json
import pandas as pd
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
        metadata: str = Form(..., description="JSON string containing customer metadata"),
        token: str = Depends(verify_token)
):
    """
    Store a new customer record with image and metadata.
    Generates embeddings and stores them in the vector database with RedisSearch indexing.
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
            store_metadata.transaction_id,
            image_data,
            store_metadata
        )

        return StoreResponse(
            status="success",
            transaction_id=store_metadata.transaction_id,
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


@router.post("/batch-store")
async def batch_store_records(
        # This would need a custom request model for batch operations
        # For now, keeping it simple with form data
        token: str = Depends(verify_token)
):
    """
    Batch store multiple customer records for better performance.
    """
    try:
        # This would be implemented to handle multiple files and metadata
        # For now, returning a placeholder response
        return {
            "status": "success",
            "message": "Batch store endpoint - implementation pending",
            "note": "Use individual store endpoint for now"
        }
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
        metadata: str = Form(..., description="JSON string containing customer metadata"),
        threshold: Optional[float] = Form(default=settings.default_similarity_threshold,
                                          description="Similarity threshold (0.0-1.0)"),
        limit: Optional[int] = Form(default=None, description="Maximum results"),
        token: str = Depends(verify_token)
):
    """
    Search for matching customer records based on image similarity.
    Uses RedisSearch with vector embeddings and configurable similarity thresholds.
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

        # Parse search metadata
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

        # Search for similar customers with metadata filtering
        results, query_metadata = await dedup_service.search_similar_customers(
            image_data,
            search_metadata,
            threshold or settings.default_similarity_threshold,
            limit
        )

        return SearchResponse(
            status="success",
            total_matches=len(results),
            metadata=query_metadata,
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


@router.post("/search-flexible")
async def search_with_flexible_filters(
        image: UploadFile = File(..., description="Query image for similarity search"),
        threshold: float = Form(default=settings.default_similarity_threshold,
                                description="Similarity threshold (0.0-1.0)"),
        limit: Optional[int] = Form(default=None, description="Maximum results"),
        id_type: Optional[str] = Form(default=None, description="Filter by ID type"),
        id_number: Optional[str] = Form(default=None, description="Filter by ID number"),
        msisdn: Optional[str] = Form(default=None, description="Filter by MSISDN"),
        token: str = Depends(verify_token)
):
    """
    Search with flexible metadata filtering options.
    Allows filtering by any combination of id_type, id_number, and msisdn.
    """
    try:
        # Validate image and parameters (similar to above)
        if image.content_type not in settings.allowed_image_types:
            raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_REQUEST", "message": f"Unsupported image format: {image.content_type}"}})

        if threshold < 0.0 or threshold > 1.0:
            raise HTTPException(status_code=400, detail={"error": {"code": "INVALID_REQUEST", "message": "Threshold must be between 0.0 and 1.0"}})

        # Read image data
        image_data = await image.read()

        # Search with flexible filters
        results = await dedup_service.search_with_flexible_filters(
            image_data, threshold, limit, id_type, id_number, msisdn
        )

        return {
            "status": "success",
            "total_matches": len(results),
            "filters_applied": {
                "id_type": id_type,
                "id_number": id_number,
                "msisdn": msisdn,
                "threshold": threshold
            },
            "results": [result.dict() for result in results]
        }

    except DedupException as e:
        raise create_http_exception(e)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Internal server error", "details": str(e)}}
        )


@router.post("/purge", response_model=PurgeResponse)
async def purge_record(
        request: PurgeRequest,
        token: str = Depends(verify_token)
):
    """
    Delete a customer record from the system.
    Removes both vector embeddings and metadata from RedisSearch index.
    """
    try:
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


@router.post("/generate-fraud-report")
async def generate_fraud_report(
        # CSV file upload for KYC data
        kyc_file: UploadFile = File(..., description="KYC CSV file"),
        similarity_threshold: float = Form(default=0.6, description="Similarity threshold for grouping"),
        output_dir: str = Form(default="./reports", description="Output directory for reports"),
        token: str = Depends(verify_token)
):
    """
    Generate comprehensive fraud detection report from uploaded KYC data.
    This replicates the functionality from your notebook.
    """
    try:
        # Read CSV file
        if not kyc_file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_REQUEST", "message": "File must be a CSV"}}
            )

        # Read CSV content
        csv_content = await kyc_file.read()

        # This would process the CSV and generate similarity groups
        # For now, returning a placeholder structure
        return {
            "status": "success",
            "message": "Fraud report generation initiated",
            "note": "Full implementation requires CSV processing and similarity grouping",
            "parameters": {
                "similarity_threshold": similarity_threshold,
                "output_dir": output_dir,
                "filename": kyc_file.filename
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "INTERNAL_ERROR", "message": str(e)}}
        )


@router.get("/statistics")
async def get_database_statistics(token: str = Depends(verify_token)):
    """
    Get Redis database and RedisSearch index statistics.
    """
    try:
        stats = await dedup_service.get_redis_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/health")
async def health_check():
    """Check service health status including RedisSearch"""
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