from typing import List, Dict, Optional
import logging
from app.models.requests import StoreMetadata, SearchMetadata
from app.models.responses import SearchResult
from app.services.embedding import embedding_service
from app.services.redis_service import redis_service
from app.utils.exceptions import DedupException, CustomerNotFoundError

logger = logging.getLogger(__name__)


class DeduplicationService:
    """Main service orchestrating the deduplication workflow"""

    def __init__(self):
        self.embedding_service = embedding_service
        self.redis_service = redis_service
        logger.info("DeduplicationService initialized")

    async def store_customer(
        self, transaction_id: str, image_data: bytes, metadata: StoreMetadata
    ) -> bool:
        """Store customer record with image embedding"""
        try:
            logger.info(f"Storing customer with transaction_id: {transaction_id}")

            # Check if customer already exists
            if await self.redis_service.customer_exists(transaction_id):
                logger.warning(f"Customer {transaction_id} already exists")
                raise DedupException(
                    f"Customer {transaction_id} already exists", "CUSTOMER_EXISTS"
                )

            # Generate embedding from image
            logger.debug(f"Generating embedding for transaction_id: {transaction_id}")
            embedding = await self.embedding_service.generate_embedding(image_data)
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")

            # Prepare metadata for storage - convert StoreMetadata to dict
            metadata_dict = {
                "transaction_id": transaction_id,  # Include transaction_id in stored metadata
                "msisdn": metadata.msisdn,
                "created_on": metadata.created_on,
                "id_type": metadata.id_type,
                "id_number": metadata.id_number,
            }

            # Store in Redis
            logger.debug(
                f"Storing vector and metadata for transaction_id: {transaction_id}"
            )
            await self.redis_service.store_vector(
                transaction_id, embedding, metadata_dict
            )

            logger.info(f"Successfully stored customer: {transaction_id}")
            return True

        except DedupException:
            # Re-raise DedupExceptions without wrapping
            raise
        except Exception as e:
            logger.error(f"Failed to store customer {transaction_id}: {str(e)}")
            raise DedupException(f"Failed to store customer: {str(e)}")

    async def search_similar_customers(
        self, image_data: bytes, threshold: float, limit: Optional[int] = None
    ) -> List[SearchResult]:
        """Search for similar customers based on image"""
        try:
            logger.info(
                f"Searching for similar customers with threshold: {threshold}, limit: {limit}"
            )

            # Generate embedding from query image
            logger.debug("Generating embedding for query image")
            query_embedding = await self.embedding_service.generate_embedding(
                image_data
            )
            logger.debug(
                f"Generated query embedding with dimension: {len(query_embedding)}"
            )

            # Set default limit if not provided
            search_limit = limit if limit is not None else 100

            # Search for similar vectors
            logger.debug(
                f"Searching Redis for similar vectors with limit: {search_limit}"
            )
            results = await self.redis_service.search_similar_vectors(
                query_embedding, threshold, search_limit
            )

            logger.info(f"Found {len(results)} similar customers")

            # Convert to SearchResult objects
            search_results = []
            for result in results:
                try:
                    # Extract stored metadata
                    stored_metadata = result["metadata"]
                    logger.debug(
                        f"Processing result for customer: {result.get('customer_id', 'unknown')}"
                    )

                    # Create StoreMetadata from stored metadata
                    # Handle both old format (customer_id) and new format (transaction_id)
                    store_metadata = StoreMetadata(
                        msisdn=stored_metadata.get("msisdn", ""),
                        created_on=stored_metadata.get("created_on"),
                        id_type=stored_metadata.get("id_type", ""),
                        id_number=stored_metadata.get("id_number", ""),
                    )

                    search_result = SearchResult(
                        similarity_score=result["similarity_score"],
                        metadata=store_metadata,
                    )
                    search_results.append(search_result)

                except Exception as e:
                    logger.warning(f"Failed to process search result: {str(e)}")
                    # Continue processing other results
                    continue

            logger.info(f"Successfully processed {len(search_results)} search results")
            return search_results

        except Exception as e:
            logger.error(f"Failed to search customers: {str(e)}")
            raise DedupException(f"Failed to search customers: {str(e)}")

    async def purge_customer(self, transaction_id: str) -> bool:
        """Delete customer record"""
        try:
            logger.info(f"Purging customer with transaction_id: {transaction_id}")

            result = await self.redis_service.delete_customer(transaction_id)

            if result:
                logger.info(f"Successfully purged customer: {transaction_id}")
            else:
                logger.warning(f"Failed to purge customer: {transaction_id}")

            return result

        except CustomerNotFoundError:
            logger.warning(f"Customer not found for purge: {transaction_id}")
            # Re-raise CustomerNotFoundError without wrapping
            raise
        except Exception as e:
            logger.error(f"Failed to purge customer {transaction_id}: {str(e)}")
            raise DedupException(f"Failed to purge customer: {str(e)}")

    async def customer_exists(self, transaction_id: str) -> bool:
        """Check if customer exists"""
        try:
            logger.debug(f"Checking if customer exists: {transaction_id}")
            exists = await self.redis_service.customer_exists(transaction_id)
            logger.debug(f"Customer {transaction_id} exists: {exists}")
            return exists
        except Exception as e:
            logger.error(
                f"Failed to check customer existence {transaction_id}: {str(e)}"
            )
            raise DedupException(f"Failed to check customer existence: {str(e)}")

    async def get_customer_count(self) -> int:
        """Get total number of stored customers"""
        try:
            # This would need to be implemented in redis_service
            # For now, return 0 as placeholder
            logger.debug("Getting customer count")
            return 0
        except Exception as e:
            logger.error(f"Failed to get customer count: {str(e)}")
            raise DedupException(f"Failed to get customer count: {str(e)}")

    async def health_check(self) -> Dict[str, bool]:
        """Check service health"""
        try:
            logger.debug("Performing health check")

            # Check Redis health
            redis_healthy = await self.redis_service.health_check()
            logger.debug(f"Redis healthy: {redis_healthy}")

            # Check embedding service status
            embedding_status = self.embedding_service.get_model_status()
            embedding_healthy = embedding_status.get("initialized", False)
            logger.debug(f"Embedding service healthy: {embedding_healthy}")

            # Overall health
            overall_healthy = redis_healthy and embedding_healthy

            health_status = {
                "redis": redis_healthy,
                "embedding": embedding_healthy,
                "overall": overall_healthy,
            }

            logger.info(f"Health check completed: {health_status}")
            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"redis": False, "embedding": False, "overall": False}

    async def get_service_statistics(self) -> Dict:
        """Get service statistics for monitoring"""
        try:
            logger.debug("Getting service statistics")

            health_status = await self.health_check()
            customer_count = await self.get_customer_count()
            embedding_status = self.embedding_service.get_model_status()

            stats = {
                "health": health_status,
                "customer_count": customer_count,
                "embedding_service": embedding_status,
                "service_info": {"name": "DeduplicationService", "version": "1.0.0"},
            }

            logger.debug(f"Service statistics: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get service statistics: {str(e)}")
            return {
                "error": str(e),
                "health": {"redis": False, "embedding": False, "overall": False},
            }


# Global instance
dedup_service = DeduplicationService()
