from typing import List, Dict, Optional
from app.models.requests import StoreMetadata, SearchMetadata
from app.models.responses import SearchResult
from app.services.embedding import embedding_service
from app.services.redis_service import redis_service
from app.utils.exceptions import DedupException, CustomerNotFoundError


class DeduplicationService:
    """Main service orchestrating the deduplication workflow"""

    def __init__(self):
        self.embedding_service = embedding_service
        self.redis_service = redis_service

    async def store_customer(
        self, transaction_id: str, image_data: bytes, metadata: StoreMetadata
    ) -> bool:
        """Store customer record with image embedding"""
        try:
            # Check if customer already exists
            if await self.redis_service.customer_exists(transaction_id):
                raise DedupException(
                    f"Customer {transaction_id} already exists", "CUSTOMER_EXISTS"
                )

            # Generate embedding from image
            embedding = await self.embedding_service.generate_embedding(image_data)

            # Store in Redis - convert StoreMetadata to dict for storage
            metadata_dict = {
                "transaction_id": transaction_id,  # Include transaction_id in stored metadata
                "msisdn": metadata.msisdn,
                "created_on": metadata.created_on,
                "id_type": metadata.id_type,
                "id_number": metadata.id_number,
            }

            await self.redis_service.store_vector(
                transaction_id, embedding, metadata_dict
            )

            return True

        except DedupException:
            raise
        except Exception as e:
            raise DedupException(f"Failed to store customer: {str(e)}")

    async def search_similar_customers(
        self, image_data: bytes, threshold: float, limit: Optional[int] = None
    ) -> List[SearchResult]:
        """Search for similar customers based on image"""
        try:
            # Generate embedding from query image
            query_embedding = await self.embedding_service.generate_embedding(
                image_data
            )

            # Search for similar vectors
            results = await self.redis_service.search_similar_vectors(
                query_embedding, threshold, limit or 100  # Use default if None
            )

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                # Create StoreMetadata from stored metadata
                stored_metadata = result["metadata"]
                store_metadata = StoreMetadata(
                    msisdn=stored_metadata.get("msisdn", ""),
                    created_on=stored_metadata.get("created_on"),
                    id_type=stored_metadata.get("id_type", ""),
                    id_number=stored_metadata.get("id_number", ""),
                )

                search_result = SearchResult(
                    similarity_score=result["similarity_score"], metadata=store_metadata
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            raise DedupException(f"Failed to search customers: {str(e)}")

    async def purge_customer(self, transaction_id: str) -> bool:
        """Delete customer record"""
        try:
            return await self.redis_service.delete_customer(transaction_id)
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
                "overall": redis_healthy,
            }
        except Exception:
            return {"redis": False, "embedding": False, "overall": False}


# Global instance
dedup_service = DeduplicationService()
