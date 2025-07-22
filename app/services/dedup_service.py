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

    def store_customer(self, customer_id: str, image_data: bytes, metadata: CustomerMetadata) -> bool:
        """Store customer record with image embedding"""
        try:
            if self.redis_service.customer_exists(customer_id):
                raise DedupException(f"Customer {customer_id} already exists", "CUSTOMER_EXISTS")

            embedding = self.embedding_service.generate_embedding(image_data)
            self.redis_service.store_vector(customer_id, embedding, metadata)
            return True

        except DedupException:
            raise
        except Exception as e:
            raise DedupException(f"Failed to store customer: {str(e)}")

    def search_similar_customers(self, image_data: bytes, threshold: float, limit: int) -> List[SearchResult]:
        """Search for similar customers based on image"""
        try:
            query_embedding = self.embedding_service.generate_embedding(image_data)
            results = self.redis_service.search_similar_vectors(query_embedding, threshold, limit)

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

    def purge_customer(self, customer_id: str) -> bool:
        """Delete customer record"""
        try:
            return self.redis_service.delete_customer(customer_id)
        except CustomerNotFoundError:
            raise
        except Exception as e:
            raise DedupException(f"Failed to purge customer: {str(e)}")

    def health_check(self) -> Dict[str, bool]:
        """Check service health"""
        try:
            redis_healthy = self.redis_service.health_check()
            return {
                "redis": redis_healthy,
                "embedding": True,
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