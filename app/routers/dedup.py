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
    """Store a new customer record with image and metadata"""
    try:
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

        image_data = await image.read()

        dedup_service.store_customer(
            customer_metadata.transaction_id,
            image_data,
            customer_metadata
        )

        return StoreResponse(
            status="success",
            transaction_id=customer_metadata.transaction_id,
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
        threshold: Optional[float] = Form(default=settings.default_similarity_threshold),
        limit: Optional[int] = Form(default=10),
        token: str = Depends(verify_token)
):
    """Search for matching customer records based on image similarity"""
    try:
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

        image_data = await image.read()

        results = dedup_service.search_similar_customers(
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
    """Delete a customer record from the system"""
    try:
        dedup_service.purge_customer(request.transaction_id)

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
        health_status = dedup_service.health_check()
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

# # Add this to your app/routers/dedup.py

# @router.post("/batch/analyze", response_model=dict)
# async def batch_analyze_duplicates(
#         threshold: Optional[float] = Form(default=settings.default_similarity_threshold),
#         limit: Optional[int] = Form(default=500),
#         token: str = Depends(verify_token)
# ):
#     """
#     Perform batch analysis of all stored records to find duplicate groups.
#     Similar to the notebook's comprehensive grouping approach.
#     """
#     try:
#         # Validate parameters
#         if threshold < 0.0 or threshold > 1.0:
#             raise HTTPException(
#                 status_code=400,
#                 detail={
#                     "error": {
#                         "code": "INVALID_REQUEST",
#                         "message": "Threshold must be between 0.0 and 1.0",
#                         "details": None
#                     }
#                 }
#             )

#         # Perform batch similarity analysis
#         groups = await redis_service.batch_search_all(threshold, limit)
        
#         # Generate report similar to notebook
#         report_data = await _generate_batch_report(groups)
        
#         return {
#             "status": "success",
#             "total_groups": len(groups),
#             "analysis_results": groups,
#             "summary": report_data
#         }

#     except DedupException as e:
#         raise create_http_exception(e)
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "error": {
#                     "code": "INTERNAL_ERROR",
#                     "message": "Internal server error",
#                     "details": str(e)
#                 }
#             }
#         )


# async def _generate_batch_report(groups: Dict) -> Dict:
#     """Generate summary report from batch analysis results"""
#     total_groups = len(groups)
#     total_records_in_groups = sum(len(group_data["similar_customers"]) + 1 for group_data in groups.values())
    
#     # Count potential fraud groups (groups with multiple unique IDs)
#     # Note: This would require metadata analysis which isn't fully implemented here
#     # You'd need to fetch customer metadata and check for unique ID numbers per group
    
#     return {
#         "total_duplicate_groups": total_groups,
#         "total_records_in_groups": total_records_in_groups,
#         "average_group_size": total_records_in_groups / total_groups if total_groups > 0 else 0,
#         "analysis_threshold": 0.6  
#     }


# @router.get("/batch/report")
# async def generate_dedup_report(
#         token: str = Depends(verify_token)
# ):
#     """
#     Generate a comprehensive deduplication report similar to the notebook output.
#     This would integrate with your existing customer metadata.
#     """
#     try:
#         # This would require integration with your customer database
#         # to fetch full metadata and generate the detailed fraud report
#         # similar to what's done in the notebook
        
#         return {
#             "status": "success",
#             "message": "Report generation endpoint - requires customer metadata integration",
#             "note": "This endpoint would generate the detailed fraud analysis report similar to the notebook"
#         }
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "error": {
#                     "code": "INTERNAL_ERROR", 
#                     "message": "Failed to generate report",
#                     "details": str(e)
#                 }
#             }
#         )
