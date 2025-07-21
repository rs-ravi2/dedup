import numpy as np
from typing import List
from PIL import Image
import io
from app.config import settings
from app.utils.exceptions import VectorServiceError


class EmbeddingService:
    """Service for generating image embeddings"""

    def __init__(self):
        self.dimension = settings.vector_dimension

    async def generate_embedding(self, image_data: bytes) -> List[float]:
        """
        Generate embedding from image data
        Currently returns a random vector - will be replaced with real model
        """
        try:
            # Validate image
            self._validate_image(image_data)

            # TODO: Replace with real model inference
            # For now, return a random normalized vector
            vector = np.random.random(self.dimension).astype(np.float32)
            # Normalize the vector
            vector = vector / np.linalg.norm(vector)

            return vector.tolist()

        except Exception as e:
            raise VectorServiceError(f"Failed to generate embedding: {str(e)}")

    def _validate_image(self, image_data: bytes) -> None:
        """Validate image format and size"""
        try:
            # Check file size
            if len(image_data) > settings.max_file_size:
                raise VectorServiceError("Image file too large")

            # Validate image format
            image = Image.open(io.BytesIO(image_data))

            # Check if image format is supported
            if image.format.lower() not in ['jpeg', 'jpg', 'png']:
                raise VectorServiceError("Unsupported image format")

            # Basic image validation
            if image.size[0] < 50 or image.size[1] < 50:
                raise VectorServiceError("Image too small")

        except Exception as e:
            if isinstance(e, VectorServiceError):
                raise
            raise VectorServiceError(f"Invalid image format: {str(e)}")

    def calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)

            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)

        except Exception as e:
            raise VectorServiceError(f"Failed to calculate similarity: {str(e)}")


# Global instance
embedding_service = EmbeddingService()