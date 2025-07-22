import redis
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.config import settings
from app.models.requests import StoreMetadata, SearchMetadata
from app.utils.exceptions import VectorServiceError, CustomerNotFoundError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import redis
import json
import numpy as np
from typing import List, Dict, Optional
from app.config import settings
from app.models.requests import CustomerMetadata
from app.utils.exceptions import VectorServiceError, CustomerNotFoundError
import asyncio


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

    def store_vector(self, customer_id: str, vector: List[float], metadata: CustomerMetadata) -> bool:
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

    def search_similar_vectors(self, query_vector: List[float], threshold: float, limit: int) -> List[Dict]:
        """Search for similar vectors using brute force similarity"""
        try:
            results = []
            vector_keys = self.redis_client.keys(f"{self.vector_key_prefix}*")

            for vector_key in vector_keys:
                customer_id = vector_key.replace(self.vector_key_prefix, "")
                stored_vector_data = self.redis_client.get(vector_key)

                if not stored_vector_data:
                    continue

                stored_vector = json.loads(stored_vector_data)
                similarity = self._calculate_cosine_similarity(query_vector, stored_vector)

                if similarity >= threshold:
                    metadata_key = f"{self.metadata_key_prefix}{customer_id}"
                    metadata_data = self.redis_client.get(metadata_key)

                    if metadata_data:
                        metadata = json.loads(metadata_data)
                        results.append({
                            "customer_id": customer_id,
                            "similarity_score": similarity,
                            "metadata": metadata
                        })

            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:limit]

        except Exception as e:
            raise VectorServiceError(f"Failed to search vectors: {str(e)}")

    def delete_customer(self, customer_id: str) -> bool:
        """Delete customer vector and metadata"""
        try:
            vector_key = f"{self.vector_key_prefix}{customer_id}"
            metadata_key = f"{self.metadata_key_prefix}{customer_id}"

            if not self.redis_client.exists(vector_key):
                raise CustomerNotFoundError(customer_id)

            pipe = self.redis_client.pipeline()
            pipe.delete(vector_key)
            pipe.delete(metadata_key)
            result = pipe.execute()

            return all(result)

        except CustomerNotFoundError:
            raise
        except Exception as e:
            raise VectorServiceError(f"Failed to delete customer: {str(e)}")

    def customer_exists(self, customer_id: str) -> bool:
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

    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            return self.redis_client.ping()
        except Exception:
            return False


# Global instance
redis_service = RedisVectorService()

#
# class RedisVectorService:
#     """Enhanced Redis-based vector storage with RedisSearch support"""
#
#     def __init__(self):
#         self.redis_client = redis.Redis.from_url(
#             settings.redis_url,
#             password=settings.redis_password,
#             db=settings.redis_db,
#             decode_responses=False  # Important for binary vector data
#         )
#         self.vector_key_prefix = "emb:"
#         self.metadata_key_prefix = "meta:"
#         self.index_name = "embedding_index"
#         self.vector_field_name = "embeddings"
#         self.batch_size = 1000
#
#         # Initialize RedisSearch index
#         self._create_vector_index()
#
#     def _create_vector_index(self) -> None:
#         """Create RedisSearch index with vector similarity support"""
#         try:
#             # Check if index already exists
#             try:
#                 self.redis_client.execute_command("FT.INFO", self.index_name)
#                 logger.info(f"Index {self.index_name} already exists")
#                 return
#             except redis.exceptions.ResponseError:
#                 # Index doesn't exist, create it
#                 pass
#
#             # Create the index with vector field and metadata fields
#             self.redis_client.execute_command(
#                 "FT.CREATE", self.index_name,
#                 "ON", "HASH",
#                 "PREFIX", "1", self.vector_key_prefix,
#                 "SCHEMA",
#                 # Vector field for similarity search
#                 self.vector_field_name, "AS", self.vector_field_name, "VECTOR", "FLAT", "6",
#                 "TYPE", "FLOAT32",
#                 "DIM", settings.vector_dimension,
#                 "DISTANCE_METRIC", "COSINE",
#                 # Metadata fields for filtering
#                 "transaction_id", "AS", "transaction_id", "TEXT",
#                 "msisdn", "AS", "msisdn", "TEXT",
#                 "id_type", "AS", "id_type", "TEXT",
#                 "id_number", "AS", "id_number", "TEXT",
#                 "created_on", "AS", "created_on", "TEXT"
#             )
#             logger.info(f"Created RedisSearch index: {self.index_name}")
#
#         except redis.exceptions.ResponseError as e:
#             if "Index already exists" in str(e):
#                 logger.info(f"Index {self.index_name} already exists")
#             else:
#                 logger.error(f"Error creating RedisSearch index: {e}")
#                 raise VectorServiceError(f"Failed to create search index: {str(e)}")
#
#     async def store_vector(self, transaction_id: str, vector: List[float], metadata: StoreMetadata) -> bool:
#         """Store vector and metadata in Redis with RedisSearch indexing"""
#         try:
#             vector_key = f"{self.vector_key_prefix}{transaction_id}"
#
#             # Convert vector to binary format for RedisSearch
#             vector_bytes = np.array(vector, dtype=np.float32).tobytes()
#
#             # Prepare hash fields for RedisSearch
#             hash_data = {
#                 self.vector_field_name: vector_bytes,
#                 "transaction_id": metadata.transaction_id,
#                 "msisdn": metadata.msisdn,
#                 "id_type": metadata.id_type,
#                 "id_number": metadata.id_number,
#                 "created_on": metadata.created_on or "",
#                 "metadata_json": metadata.model_dump_json()  # Keep full metadata as JSON
#             }
#
#             # Store in Redis hash (automatically indexed by RedisSearch)
#             self.redis_client.hset(vector_key, mapping=hash_data)
#             logger.info(f"Stored vector for transaction_id: {transaction_id}")
#
#             return True
#
#         except Exception as e:
#             logger.error(f"Failed to store vector for {transaction_id}: {str(e)}")
#             raise VectorServiceError(f"Failed to store vector: {str(e)}")
#
#     async def search_similar_vectors(
#             self,
#             query_vector: List[float],
#             threshold: float,
#             limit: Optional[int] = None,
#             metadata_filters: Optional[SearchMetadata] = None
#     ) -> List[Dict]:
#         """Search for similar vectors using RedisSearch with optional metadata filtering"""
#         try:
#             # Convert query vector to binary format
#             query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
#
#             # Build the search query
#             base_query = f"*=>[KNN {limit or 100} @{self.vector_field_name} $vec AS score]"
#
#             # Add metadata filters if provided
#             if metadata_filters:
#                 filter_conditions = []
#                 if metadata_filters.id_type:
#                     filter_conditions.append(f"@id_type:{metadata_filters.id_type}")
#                 if metadata_filters.id_number:
#                     filter_conditions.append(f"@id_number:{metadata_filters.id_number}")
#
#                 if filter_conditions:
#                     filter_query = " ".join(filter_conditions)
#                     base_query = f"({filter_query})=>[KNN {limit or 100} @{self.vector_field_name} $vec AS score]"
#
#             # Execute the search
#             search_params = [
#                 "FT.SEARCH", self.index_name,
#                 base_query,
#                 "SORTBY", "score", "ASC",
#                 "RETURN", "3", "score", "transaction_id", "metadata_json",
#                 "LIMIT", "0", str(limit or 100),
#                 "PARAMS", "2", "vec", query_vector_bytes,
#                 "DIALECT", "2"
#             ]
#
#             result = self.redis_client.execute_command(*search_params)
#
#             # Parse results
#             results = []
#             if len(result) > 1:
#                 # result[0] is the count, result[1:] are the actual results
#                 for i in range(1, len(result), 2):
#                     doc_id = result[i].decode() if isinstance(result[i], bytes) else result[i]
#                     doc_fields = result[i + 1]
#
#                     # Extract fields from the result
#                     fields = {}
#                     for j in range(0, len(doc_fields), 2):
#                         key = doc_fields[j].decode() if isinstance(doc_fields[j], bytes) else doc_fields[j]
#                         value = doc_fields[j + 1]
#                         if isinstance(value, bytes):
#                             value = value.decode()
#                         fields[key] = value
#
#                     # Calculate similarity score and apply threshold
#                     if 'score' in fields:
#                         cosine_distance = float(fields['score'])
#                         similarity_score = 1 - cosine_distance
#
#                         if similarity_score >= threshold:
#                             # Parse metadata
#                             metadata = json.loads(fields.get('metadata_json', '{}'))
#
#                             results.append({
#                                 "transaction_id": fields.get('transaction_id', doc_id.replace(self.vector_key_prefix, "")),
#                                 "similarity_score": similarity_score,
#                                 "metadata": metadata
#                             })
#
#             # Sort by similarity score (descending) and apply limit
#             results.sort(key=lambda x: x["similarity_score"], reverse=True)
#             if limit:
#                 results = results[:limit]
#
#             logger.info(f"Found {len(results)} similar vectors above threshold {threshold}")
#             return results
#
#         except Exception as e:
#             logger.error(f"Failed to search vectors: {str(e)}")
#             raise VectorServiceError(f"Failed to search vectors: {str(e)}")
#
#     async def search_with_metadata_filters(
#             self,
#             query_vector: List[float],
#             threshold: float,
#             limit: Optional[int] = None,
#             id_type: Optional[str] = None,
#             id_number: Optional[str] = None,
#             msisdn: Optional[str] = None
#     ) -> List[Dict]:
#         """Search with flexible metadata filtering"""
#         try:
#             query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
#
#             # Build filter conditions
#             filter_conditions = []
#             if id_type:
#                 filter_conditions.append(f"@id_type:{id_type}")
#             if id_number:
#                 filter_conditions.append(f"@id_number:{id_number}")
#             if msisdn:
#                 filter_conditions.append(f"@msisdn:{msisdn}")
#
#             # Construct query
#             if filter_conditions:
#                 filter_query = " ".join(filter_conditions)
#                 base_query = f"({filter_query})=>[KNN {limit or 100} @{self.vector_field_name} $vec AS score]"
#             else:
#                 base_query = f"*=>[KNN {limit or 100} @{self.vector_field_name} $vec AS score]"
#
#             # Execute search (similar to above method)
#             search_params = [
#                 "FT.SEARCH", self.index_name,
#                 base_query,
#                 "SORTBY", "score", "ASC",
#                 "RETURN", "3", "score", "transaction_id", "metadata_json",
#                 "LIMIT", "0", str(limit or 100),
#                 "PARAMS", "2", "vec", query_vector_bytes,
#                 "DIALECT", "2"
#             ]
#
#             result = self.redis_client.execute_command(*search_params)
#
#             # Parse and filter results (same logic as above)
#             results = []
#             if len(result) > 1:
#                 for i in range(1, len(result), 2):
#                     doc_id = result[i].decode() if isinstance(result[i], bytes) else result[i]
#                     doc_fields = result[i + 1]
#
#                     fields = {}
#                     for j in range(0, len(doc_fields), 2):
#                         key = doc_fields[j].decode() if isinstance(doc_fields[j], bytes) else doc_fields[j]
#                         value = doc_fields[j + 1]
#                         if isinstance(value, bytes):
#                             value = value.decode()
#                         fields[key] = value
#
#                     if 'score' in fields:
#                         cosine_distance = float(fields['score'])
#                         similarity_score = 1 - cosine_distance
#
#                         if similarity_score >= threshold:
#                             metadata = json.loads(fields.get('metadata_json', '{}'))
#                             results.append({
#                                 "transaction_id": fields.get('transaction_id', doc_id.replace(self.vector_key_prefix, "")),
#                                 "similarity_score": similarity_score,
#                                 "metadata": metadata
#                             })
#
#             results.sort(key=lambda x: x["similarity_score"], reverse=True)
#             if limit:
#                 results = results[:limit]
#
#             return results
#
#         except Exception as e:
#             logger.error(f"Failed to search with metadata filters: {str(e)}")
#             raise VectorServiceError(f"Failed to search with metadata filters: {str(e)}")
#
#     async def batch_store_vectors(self, batch_data: List[Tuple[str, List[float], StoreMetadata]]) -> bool:
#         """Batch store multiple vectors for better performance"""
#         try:
#             pipe = self.redis_client.pipeline()
#
#             for transaction_id, vector, metadata in batch_data:
#                 vector_key = f"{self.vector_key_prefix}{transaction_id}"
#                 vector_bytes = np.array(vector, dtype=np.float32).tobytes()
#
#                 hash_data = {
#                     self.vector_field_name: vector_bytes,
#                     "transaction_id": metadata.transaction_id,
#                     "msisdn": metadata.msisdn,
#                     "id_type": metadata.id_type,
#                     "id_number": metadata.id_number,
#                     "created_on": metadata.created_on or "",
#                     "metadata_json": metadata.model_dump_json()
#                 }
#
#                 pipe.hset(vector_key, mapping=hash_data)
#
#             pipe.execute()
#             logger.info(f"Batch stored {len(batch_data)} vectors")
#             return True
#
#         except Exception as e:
#             logger.error(f"Failed to batch store vectors: {str(e)}")
#             raise VectorServiceError(f"Failed to batch store vectors: {str(e)}")
#
#     async def delete_customer(self, transaction_id: str) -> bool:
#         """Delete customer vector and metadata"""
#         try:
#             vector_key = f"{self.vector_key_prefix}{transaction_id}"
#
#             # Check if customer exists
#             if not self.redis_client.exists(vector_key):
#                 raise CustomerNotFoundError(transaction_id)
#
#             # Delete the hash (automatically removed from RedisSearch index)
#             result = self.redis_client.delete(vector_key)
#             logger.info(f"Deleted customer: {transaction_id}")
#             return bool(result)
#
#         except CustomerNotFoundError:
#             raise
#         except Exception as e:
#             logger.error(f"Failed to delete customer {transaction_id}: {str(e)}")
#             raise VectorServiceError(f"Failed to delete customer: {str(e)}")
#
#     async def customer_exists(self, transaction_id: str) -> bool:
#         """Check if customer exists in the database"""
#         try:
#             vector_key = f"{self.vector_key_prefix}{transaction_id}"
#             return bool(self.redis_client.exists(vector_key))
#         except Exception as e:
#             logger.error(f"Failed to check customer existence: {str(e)}")
#             raise VectorServiceError(f"Failed to check customer existence: {str(e)}")
#
#     async def get_index_info(self) -> Dict:
#         """Get information about the RedisSearch index"""
#         try:
#             info = self.redis_client.execute_command("FT.INFO", self.index_name)
#             return {"index_info": info}
#         except Exception as e:
#             logger.error(f"Failed to get index info: {str(e)}")
#             return {"error": str(e)}
#
#     async def health_check(self) -> bool:
#         """Check Redis connection health"""
#         try:
#             ping_result = self.redis_client.ping()
#             # Also check if RedisSearch is available
#             try:
#                 self.redis_client.execute_command("FT.INFO", self.index_name)
#                 redisearch_available = True
#             except:
#                 redisearch_available = False
#
#             return ping_result and redisearch_available
#         except Exception:
#             return False
#
#     def count_keys(self) -> int:
#         """Count total keys in Redis"""
#         return self.redis_client.dbsize()
#
#
# # Global instance
# redis_service = RedisVectorService()