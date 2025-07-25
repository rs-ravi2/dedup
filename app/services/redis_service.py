import redis.asyncio as redis
import json
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union
from app.config import settings
from app.utils.exceptions import VectorServiceError, CustomerNotFoundError

logger = logging.getLogger(__name__)


class RedisVectorService:
    """Redis-based vector storage and similarity search service"""

    def __init__(self):
        self.redis_client = None
        self.vector_key_prefix = "vector:"
        self.metadata_key_prefix = "metadata:"
        self.stats_key = "dedup:stats"
        self._connection_pool = None
        logger.info("RedisVectorService initialized")

    async def _get_redis_client(self):
        """Get Redis client with connection pooling"""
        if self.redis_client is None:
            try:
                # Create connection pool for better performance
                self._connection_pool = redis.ConnectionPool.from_url(
                    settings.redis_url,
                    password=settings.redis_password,
                    db=settings.redis_db,
                    decode_responses=True,
                    max_connections=20,
                    retry_on_timeout=True,
                )

                self.redis_client = redis.Redis(connection_pool=self._connection_pool)

                # Test connection
                await self.redis_client.ping()
                logger.info("Redis connection established successfully")

            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise VectorServiceError(f"Redis connection failed: {str(e)}")

        return self.redis_client

    async def store_vector(
        self, transaction_id: str, vector: List[float], metadata: Union[Dict, object]
    ) -> bool:
        """Store vector and metadata in Redis"""
        try:
            logger.debug(f"Storing vector for transaction_id: {transaction_id}")

            client = await self._get_redis_client()

            # Prepare keys
            vector_key = f"{self.vector_key_prefix}{transaction_id}"
            metadata_key = f"{self.metadata_key_prefix}{transaction_id}"

            # Serialize data
            vector_data = json.dumps(vector)

            # Handle metadata - convert to dict if it's a Pydantic model
            if hasattr(metadata, "dict"):
                metadata_dict = metadata.dict()
            elif hasattr(metadata, "model_dump"):
                metadata_dict = metadata.model_dump()
            else:
                metadata_dict = metadata

            metadata_data = json.dumps(metadata_dict)

            # Use pipeline for atomic operations
            async with client.pipeline() as pipe:
                pipe.set(vector_key, vector_data)
                pipe.set(metadata_key, metadata_data)
                # Update statistics
                pipe.incr(f"{self.stats_key}:total_customers")
                pipe.hset(
                    f"{self.stats_key}:last_operation",
                    mapping={
                        "type": "store",
                        "transaction_id": transaction_id,
                        "timestamp": str(np.datetime64("now")),
                    },
                )
                await pipe.execute()

            logger.info(
                f"Successfully stored vector for transaction_id: {transaction_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store vector for {transaction_id}: {str(e)}")
            raise VectorServiceError(f"Failed to store vector: {str(e)}")

    async def search_similar_vectors(
        self, query_vector: List[float], threshold: float, limit: Optional[int] = None
    ) -> List[Dict]:
        """Search for similar vectors using brute force similarity"""
        try:
            logger.debug(
                f"Searching similar vectors with threshold: {threshold}, limit: {limit}"
            )

            client = await self._get_redis_client()
            results = []

            # Get all vector keys
            vector_keys = await client.keys(f"{self.vector_key_prefix}*")
            logger.debug(f"Found {len(vector_keys)} stored vectors to compare")

            if not vector_keys:
                logger.info("No vectors found in database")
                return results

            # Process each vector
            for vector_key in vector_keys:
                try:
                    # Extract transaction_id from key
                    transaction_id = vector_key.replace(self.vector_key_prefix, "")

                    # Get stored vector
                    stored_vector_data = await client.get(vector_key)
                    if not stored_vector_data:
                        logger.warning(f"Empty vector data for key: {vector_key}")
                        continue

                    stored_vector = json.loads(stored_vector_data)

                    # Calculate similarity
                    similarity = self._calculate_cosine_similarity(
                        query_vector, stored_vector
                    )

                    if similarity >= threshold:
                        # Get metadata
                        metadata_key = f"{self.metadata_key_prefix}{transaction_id}"
                        metadata_data = await client.get(metadata_key)

                        if metadata_data:
                            try:
                                metadata = json.loads(metadata_data)
                                results.append(
                                    {
                                        "customer_id": transaction_id,  # For backwards compatibility
                                        "transaction_id": transaction_id,  # New field
                                        "similarity_score": similarity,
                                        "metadata": metadata,
                                    }
                                )
                                logger.debug(
                                    f"Added result for {transaction_id} with similarity {similarity:.3f}"
                                )
                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"Failed to parse metadata for {transaction_id}: {str(e)}"
                                )
                                continue
                        else:
                            logger.warning(
                                f"No metadata found for transaction_id: {transaction_id}"
                            )

                except Exception as e:
                    logger.warning(f"Error processing vector {vector_key}: {str(e)}")
                    continue

            # Sort by similarity score (descending) and limit results
            results.sort(key=lambda x: x["similarity_score"], reverse=True)

            if limit is not None and limit > 0:
                results = results[:limit]

            # Update search statistics
            try:
                async with client.pipeline() as pipe:
                    pipe.incr(f"{self.stats_key}:total_searches")
                    pipe.hset(
                        f"{self.stats_key}:last_search",
                        mapping={
                            "threshold": threshold,
                            "limit": limit or "unlimited",
                            "results_count": len(results),
                            "timestamp": str(np.datetime64("now")),
                        },
                    )
                    await pipe.execute()
            except Exception as e:
                logger.warning(f"Failed to update search statistics: {str(e)}")

            logger.info(
                f"Search completed: found {len(results)} results above threshold {threshold}"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            raise VectorServiceError(f"Failed to search vectors: {str(e)}")

    async def delete_customer(self, transaction_id: str) -> bool:
        """Delete customer vector and metadata"""
        try:
            logger.debug(f"Deleting customer: {transaction_id}")

            client = await self._get_redis_client()

            vector_key = f"{self.vector_key_prefix}{transaction_id}"
            metadata_key = f"{self.metadata_key_prefix}{transaction_id}"

            # Check if customer exists
            if not await client.exists(vector_key):
                logger.warning(f"Customer {transaction_id} not found for deletion")
                raise CustomerNotFoundError(transaction_id)

            # Delete both keys atomically
            async with client.pipeline() as pipe:
                pipe.delete(vector_key)
                pipe.delete(metadata_key)
                # Update statistics
                pipe.decr(f"{self.stats_key}:total_customers")
                pipe.incr(f"{self.stats_key}:total_deletions")
                pipe.hset(
                    f"{self.stats_key}:last_operation",
                    mapping={
                        "type": "delete",
                        "transaction_id": transaction_id,
                        "timestamp": str(np.datetime64("now")),
                    },
                )
                result = await pipe.execute()

            success = all(result[:2])  # Check if both delete operations succeeded

            if success:
                logger.info(f"Successfully deleted customer: {transaction_id}")
            else:
                logger.error(f"Failed to delete customer: {transaction_id}")

            return success

        except CustomerNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete customer {transaction_id}: {str(e)}")
            raise VectorServiceError(f"Failed to delete customer: {str(e)}")

    async def customer_exists(self, transaction_id: str) -> bool:
        """Check if customer exists in the database"""
        try:
            logger.debug(f"Checking existence of customer: {transaction_id}")

            client = await self._get_redis_client()
            vector_key = f"{self.vector_key_prefix}{transaction_id}"
            exists = bool(await client.exists(vector_key))

            logger.debug(f"Customer {transaction_id} exists: {exists}")
            return exists

        except Exception as e:
            logger.error(
                f"Failed to check customer existence {transaction_id}: {str(e)}"
            )
            raise VectorServiceError(f"Failed to check customer existence: {str(e)}")

    async def get_customer_metadata(self, transaction_id: str) -> Optional[Dict]:
        """Get customer metadata by transaction_id"""
        try:
            logger.debug(f"Getting metadata for customer: {transaction_id}")

            client = await self._get_redis_client()
            metadata_key = f"{self.metadata_key_prefix}{transaction_id}"

            metadata_data = await client.get(metadata_key)
            if metadata_data:
                metadata = json.loads(metadata_data)
                logger.debug(f"Retrieved metadata for {transaction_id}")
                return metadata
            else:
                logger.warning(f"No metadata found for {transaction_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to get metadata for {transaction_id}: {str(e)}")
            raise VectorServiceError(f"Failed to get customer metadata: {str(e)}")

    async def get_customer_count(self) -> int:
        """Get total number of customers"""
        try:
            client = await self._get_redis_client()

            # Try to get from statistics first
            count = await client.get(f"{self.stats_key}:total_customers")
            if count is not None:
                return int(count)

            # Fallback: count keys manually
            vector_keys = await client.keys(f"{self.vector_key_prefix}*")
            count = len(vector_keys)

            # Update statistics
            await client.set(f"{self.stats_key}:total_customers", count)

            logger.debug(f"Total customer count: {count}")
            return count

        except Exception as e:
            logger.error(f"Failed to get customer count: {str(e)}")
            raise VectorServiceError(f"Failed to get customer count: {str(e)}")

    async def get_statistics(self) -> Dict:
        """Get Redis service statistics"""
        try:
            client = await self._get_redis_client()

            # Get all statistics
            stats = {}
            stats["total_customers"] = (
                await client.get(f"{self.stats_key}:total_customers") or 0
            )
            stats["total_searches"] = (
                await client.get(f"{self.stats_key}:total_searches") or 0
            )
            stats["total_deletions"] = (
                await client.get(f"{self.stats_key}:total_deletions") or 0
            )

            # Get last operation info
            last_operation = await client.hgetall(f"{self.stats_key}:last_operation")
            if last_operation:
                stats["last_operation"] = last_operation

            # Get last search info
            last_search = await client.hgetall(f"{self.stats_key}:last_search")
            if last_search:
                stats["last_search"] = last_search

            # Convert string numbers to integers
            for key in ["total_customers", "total_searches", "total_deletions"]:
                try:
                    stats[key] = int(stats[key])
                except (ValueError, TypeError):
                    stats[key] = 0

            logger.debug(f"Retrieved statistics: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}

    def _calculate_cosine_similarity(
        self, vector1: List[float], vector2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if len(vector1) != len(vector2):
                logger.warning(
                    f"Vector dimension mismatch: {len(vector1)} vs {len(vector2)}"
                )
                return 0.0

            v1 = np.array(vector1, dtype=np.float32)
            v2 = np.array(vector2, dtype=np.float32)

            # Handle zero vectors
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                logger.debug("Zero vector encountered in similarity calculation")
                return 0.0

            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            similarity = dot_product / (norm_v1 * norm_v2)

            # Ensure similarity is in valid range [-1, 1]
            similarity = float(np.clip(similarity, -1.0, 1.0))

            return similarity

        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            raise VectorServiceError(f"Failed to calculate similarity: {str(e)}")

    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            client = await self._get_redis_client()
            await client.ping()
            logger.debug("Redis health check passed")
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False

    async def cleanup_expired_data(self, max_age_days: int = 30) -> int:
        """Cleanup old data (placeholder for future implementation)"""
        try:
            # This could be implemented to clean up old records
            # For now, just return 0
            logger.info(f"Cleanup requested for data older than {max_age_days} days")
            return 0
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return 0

    async def close(self):
        """Close Redis connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {str(e)}")


# Global instance
redis_service = RedisVectorService()
