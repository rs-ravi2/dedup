from typing import List, Dict
from app.models.requests import CustomerMetadata
from app.models.responses import SearchResult
from app.services.embedding import embedding_service
from app.services.redis_service import redis_service
from app.utils.exceptions import DedupException, CustomerNotFoundError


class DeduplicationService:
    """Main service orchestrating the deduplication workflow"""

    def __init__(self):
        self.embedding_service = embedding_service
        self.redis_service = redis_service

    async def store_customer(self, customer_id: str, image_data: bytes, metadata: CustomerMetadata) -> bool:
        """Store customer record with image embedding"""
        try:
            # Check if customer already exists
            if await self.redis_service.customer_exists(customer_id):
                raise DedupException(f"Customer {customer_id} already exists", "CUSTOMER_EXISTS")

            # Generate embedding from image
            embedding = await self.embedding_service.generate_embedding(image_data)

            # Store in Redis
            await self.redis_service.store_vector(customer_id, embedding, metadata)

            return True

        except DedupException:
            raise
        except Exception as e:
            raise DedupException(f"Failed to store customer: {str(e)}")

    async def search_similar_customers(self, image_data: bytes, threshold: float, limit: int) -> List[SearchResult]:
        """Search for similar customers based on image"""
        try:
            # Generate embedding from query image
            query_embedding = await self.embedding_service.generate_embedding(image_data)

            # Search for similar vectors
            results = await self.redis_service.search_similar_vectors(
                query_embedding, threshold, limit
            )

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_result = SearchResult(
                    customer_id=result["customer_id"],
                    similarity_score=result["similarity_score"],
                    metadata=CustomerMetadata(**result["metadata"])
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            raise DedupException(f"Failed to search customers: {str(e)}")

    async def purge_customer(self, customer_id: str) -> bool:
        """Delete customer record"""
        try:
            return await self.redis_service.delete_customer(customer_id)
        except CustomerNotFoundError:
            raise
        except Exception as e:
            raise DedupException(f"Failed to purge customer: {str(e)}")

    async def health_check(self) -> Dict[str, bool]:
        """Check service health"""
        try:
            redis_healthy = await self.redis_service.health_check()
            return {
                "redis": redis_healthy,
                "embedding": True,  # Stub service is always healthy
                "overall": redis_healthy
            }
        except Exception:
            return {
                "redis": False,
                "embedding": False,
                "overall": False
            }


# Global instance
dedup_service = DeduplicationService()