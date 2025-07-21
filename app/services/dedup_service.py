from typing import List, Dict, Optional
from app.models.requests import StoreMetadata, SearchMetadata
from app.models.responses import SearchResult, SearchResultMetadata
from app.services.embedding import embedding_service
from app.services.redis_service import redis_service
from app.utils.exceptions import DedupException, CustomerNotFoundError


class DeduplicationService:
    """Main service orchestrating the deduplication workflow"""

    def __init__(self):
        self.embedding_service = embedding_service
        self.redis_service = redis_service

    async def store_customer(self, transaction_id: str, image_data: bytes, metadata: StoreMetadata) -> bool:
        """Store customer record with image embedding"""
        try:
            # Check if customer already exists
            if await self.redis_service.customer_exists(transaction_id):
                raise DedupException(f"Customer {transaction_id} already exists", "CUSTOMER_EXISTS")

            # Generate embedding from image
            embedding = await self.embedding_service.generate_embedding(image_data)

            # Store in Redis
            await self.redis_service.store_vector(transaction_id, embedding, metadata)

            return True

        except DedupException:
            raise
        except Exception as e:
            raise DedupException(f"Failed to store customer: {str(e)}")

    async def search_similar_customers(
            self,
            image_data: bytes,
            search_metadata: SearchMetadata,
            threshold: float,
            limit: Optional[int] = None
    ) -> tuple[List[SearchResult], SearchMetadata]:
        """Search for similar customers based on image and apply metadata filtering"""
        try:
            # Log search request for audit purposes
            print(f"Search request - Transaction ID: {search_metadata.transaction_id}, "
                  f"ID Type: {search_metadata.id_type}, ID Number: {search_metadata.id_number}")

            # Generate embedding from query image
            query_embedding = await self.embedding_service.generate_embedding(image_data)

            # Search for similar vectors
            all_results = await self.redis_service.search_similar_vectors(
                query_embedding, threshold, limit
            )

            # Apply metadata filtering based on search criteria
            filtered_results = []
            for result in all_results:
                stored_metadata = result["metadata"]

                # Filter by id_type and id_number if they match
                if (stored_metadata.get("id_type") == search_metadata.id_type and
                        stored_metadata.get("id_number") == search_metadata.id_number):
                    # Create SearchResultMetadata from stored metadata
                    result_metadata = SearchResultMetadata(
                        msisdn=stored_metadata["msisdn"],
                        created_on=stored_metadata["created_on"],
                        id_type=stored_metadata["id_type"],
                        id_number=stored_metadata["id_number"]
                    )

                    search_result = SearchResult(
                        similarity_score=result["similarity_score"],
                        metadata=result_metadata
                    )
                    filtered_results.append(search_result)

            # Log search results
            print(f"Search completed - Found {len(filtered_results)} matching records after filtering")

            # Return filtered results and echo back the search metadata
            return filtered_results, search_metadata

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