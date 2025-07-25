import redis
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.config import settings
from app.models.requests import StoreMetadata, SearchMetadata
from app.models.requests import CustomerMetadata
from app.utils.exceptions import VectorServiceError, CustomerNotFoundError
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisVectorService:
    """Redis-based vector storage and similarity search service"""

    def __init__(self):
        logger.info("Initializing RedisVectorService...")
        try:
            self.redis_client = redis.Redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=True
            )
            # Test connection
            if self.redis_client.ping():
                logger.info("Successfully connected to Redis DB.")
            else:
                logger.warning("Ping to Redis DB failed during initialization.")
        except Exception as e:
            logger.error(f"Error connecting to Redis DB: {str(e)}")
            raise VectorServiceError(f"Failed to connect to Redis: {str(e)}")
        self.vector_key_prefix = "vector:"
        self.metadata_key_prefix = "metadata:"

    def store_vector(self, transaction_id: str, vector: List[float], metadata: CustomerMetadata) -> bool:
        """Store vector and metadata in Redis"""
        try:
            logger.info(f"Storing vector for transaction_id: {transaction_id}")
            t1 = time.perf_counter()
            vector_key = f"{self.vector_key_prefix}{transaction_id}"
            vector_data = json.dumps(vector)

            metadata_key = f"{self.metadata_key_prefix}{transaction_id}"
            if hasattr(metadata, 'model_dump'):
                metadata_data = json.dumps(metadata.model_dump())
            else:
                metadata_data = json.dumps(metadata.dict())

            pipe = self.redis_client.pipeline()
            pipe.set(vector_key, vector_data)
            pipe.set(metadata_key, metadata_data)
            pipe.execute()
            logger.info(f"Stored vector and metadata for transaction_id: {transaction_id}")
            logger.info(f"Processing time for storing vector: {time.perf_counter() - t1} seconds")

            return True

        except Exception as e:
            logger.error(f"Failed to store vector for transaction_id {transaction_id}: {str(e)}")
            raise VectorServiceError(f"Failed to store vector: {str(e)}")

    def search_similar_vectors(self, query_vector: List[float], threshold: float, limit: int) -> List[Dict]:
        """Search for similar vectors using brute force similarity"""
        try:
            logger.info(f"Searching similar vectors with threshold={threshold}, limit={limit}")
            results = []
            t1 = time.perf_counter()
            vector_keys = self.redis_client.keys(f"{self.vector_key_prefix}*")
            logger.info(f"Found {len(vector_keys)} vector keys in Redis.")

            for vector_key in vector_keys:
                transaction_id = vector_key.replace(self.vector_key_prefix, "")
                stored_vector_data = self.redis_client.get(vector_key)

                if not stored_vector_data:
                    logger.warning(f"No vector data found for key: {vector_key}")
                    continue

                stored_vector = json.loads(stored_vector_data)
                similarity = self._calculate_cosine_similarity(query_vector, stored_vector)

                if similarity >= threshold:
                    metadata_key = f"{self.metadata_key_prefix}{transaction_id}"
                    metadata_data = self.redis_client.get(metadata_key)

                    if metadata_data:
                        metadata = json.loads(metadata_data)
                        results.append({
                            "transaction_id": transaction_id,
                            "similarity_score": similarity,
                            "metadata": metadata
                        })
            logger.info(f"Processing time for searching vectors: {time.perf_counter() - t1} seconds")
            logger.info(f"Found {len(results)} similar vectors above threshold.")

            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            raise VectorServiceError(f"Failed to search vectors: {str(e)}")

    def delete_customer(self, transaction_id: str) -> bool:
        """Delete customer vector and metadata"""
        try:
            logger.info(f"Deleting customer with transaction_id: {transaction_id}")
            t1 = time.perf_counter()
            vector_key = f"{self.vector_key_prefix}{transaction_id}"
            metadata_key = f"{self.metadata_key_prefix}{transaction_id}"

            if not self.redis_client.exists(vector_key):
                logger.warning(f"Customer not found for transaction_id: {transaction_id}")
                raise CustomerNotFoundError(transaction_id)

            pipe = self.redis_client.pipeline()
            pipe.delete(vector_key)
            pipe.delete(metadata_key)
            result = pipe.execute()
            logger.info(f"Deleted vector and metadata for transaction_id: {transaction_id}")
            logger.info(f"Processing time for deleting customer: {time.perf_counter() - t1} seconds")

            return all(result)

        except CustomerNotFoundError:
            logger.error(f"CustomerNotFoundError for transaction_id: {transaction_id}")
            raise
        except Exception as e:
            logger.error(f"Failed to delete customer {transaction_id}: {str(e)}")
            raise VectorServiceError(f"Failed to delete customer: {str(e)}")

    def customer_exists(self, transaction_id: str) -> bool:
        """Check if customer exists in the database"""
        try:
            vector_key = f"{self.vector_key_prefix}{transaction_id}"
            exists = bool(self.redis_client.exists(vector_key))
            logger.info(f"Customer existence check for transaction_id {transaction_id}: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Failed to check customer existence for {transaction_id}: {str(e)}")
            raise VectorServiceError(f"Failed to check customer existence: {str(e)}")

    def _calculate_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                logger.warning("Zero norm encountered in cosine similarity calculation.")
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)
            logger.debug(f"Cosine similarity calculated: {similarity}")
            return float(similarity)

        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            raise VectorServiceError(f"Failed to calculate similarity: {str(e)}")

    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            ping_result = self.redis_client.ping()
            logger.info(f"Redis health check ping result: {ping_result}")
            return ping_result
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False

# Global instance
redis_service = RedisVectorService()