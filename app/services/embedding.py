import numpy as np
from typing import List
from PIL import Image
import io
from app.config import settings
from app.utils.exceptions import VectorServiceError
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating image embeddings"""

    def __init__(self):
        self.dimension = settings.vector_dimension

    def generate_embedding(self, image_data: bytes) -> List[float]:
        """Generate embedding from image data"""
        try:
            t1 = time.perf_counter()
            self._validate_image(image_data)

            # TODO: Replace with real model inference
            vector = np.random.random(self.dimension).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            logger.info(f"Processing time for emebedding generation {time.perf_counter() - t1} seconds")
            return vector.tolist()

        except Exception as e:
            raise VectorServiceError(f"Failed to generate embedding: {str(e)}")

    def _validate_image(self, image_data: bytes) -> None:
        """Validate image format and size"""
        try:
            if len(image_data) > settings.max_file_size:
                raise VectorServiceError("Image file too large")

            image = Image.open(io.BytesIO(image_data))

            if image.format.lower() not in ["jpeg", "jpg", "png"]:
                raise VectorServiceError("Unsupported image format")

            if image.size[0] < 50 or image.size[1] < 50:
                raise VectorServiceError("Image too small")
            

        except Exception as e:
            if isinstance(e, VectorServiceError):
                raise
            raise VectorServiceError(f"Invalid image format: {str(e)}")


# Global instance
embedding_service = EmbeddingService()
