import redis
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.config import settings
from app.models.requests import CustomerMetadata
from app.utils.exceptions import VectorServiceError, CustomerNotFoundError


class RedisVectorService:
    """Redis-based vector storage and similarity search service"""

    def __init__(self):
        self.redis_client = redis.Redis.from_url(
            settings.redis_url,
            password=settings.redis_password,
            db=settings.redis_db,
            decode_responses=True
        )
        self.vector_key_prefix = "vector:"
        self.metadata_key_prefix = "metadata:"

    async def store_vector(self, customer_id: str, vector: List[float], metadata: CustomerMetadata) -> bool:
        """Store vector and metadata in Redis"""
        try:
            # Store vector
            vector_key = f"{self.vector_key_prefix}{customer_id}"
            vector_data = json.dumps(vector)

            # Store metadata
            metadata_key = f"{self.metadata_key_prefix}{customer_id}"
            metadata_data = metadata.model_dump_json()

            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.set(vector_key, vector_data)
            pipe.set(metadata_key, metadata_data)
            pipe.execute()

            return True

        except Exception as e:
            raise VectorServiceError(f"Failed to store vector: {str(e)}")

    async def search_similar_vectors(self, query_vector: List[float], threshold: float, limit: int) -> List[Dict]:
        """Search for similar vectors using brute force similarity"""
        try:
            results = []

            # Get all vector keys
            vector_keys = self.redis_client.keys(f"{self.vector_key_prefix}*")

            for vector_key in vector_keys:
                # Extract customer_id from key
                customer_id = vector_key.replace(self.vector_key_prefix, "")

                # Get stored vector
                stored_vector_data = self.redis_client.get(vector_key)
                if not stored_vector_data:
                    continue

                stored_vector = json.loads(stored_vector_data)

                # Calculate similarity
                similarity = self._calculate_cosine_similarity(query_vector, stored_vector)

                if similarity >= threshold:
                    # Get metadata
                    metadata_key = f"{self.metadata_key_prefix}{customer_id}"
                    metadata_data = self.redis_client.get(metadata_key)

                    if metadata_data:
                        metadata = json.loads(metadata_data)
                        results.append({
                            "customer_id": customer_id,
                            "similarity_score": similarity,
                            "metadata": metadata
                        })

            # Sort by similarity score (descending) and limit results
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:limit]

        except Exception as e:
            raise VectorServiceError(f"Failed to search vectors: {str(e)}")

    async def delete_customer(self, customer_id: str) -> bool:
        """Delete customer vector and metadata"""
        try:
            vector_key = f"{self.vector_key_prefix}{customer_id}"
            metadata_key = f"{self.metadata_key_prefix}{customer_id}"

            # Check if customer exists
            if not self.redis_client.exists(vector_key):
                raise CustomerNotFoundError(customer_id)

            # Delete both keys
            pipe = self.redis_client.pipeline()
            pipe.delete(vector_key)
            pipe.delete(metadata_key)
            result = pipe.execute()

            return all(result)

        except CustomerNotFoundError:
            raise
        except Exception as e:
            raise VectorServiceError(f"Failed to delete customer: {str(e)}")

    async def customer_exists(self, customer_id: str) -> bool:
        """Check if customer exists in the database"""
        try:
            vector_key = f"{self.vector_key_prefix}{customer_id}"
            return bool(self.redis_client.exists(vector_key))
        except Exception as e:
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
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)

        except Exception as e:
            raise VectorServiceError(f"Failed to calculate similarity: {str(e)}")

    async def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            return self.redis_client.ping()
        except Exception:
            return False


# Global instance
redis_service = RedisVectorService()