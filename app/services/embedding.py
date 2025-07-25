import numpy as np
import os
import logging
import zipfile
import urllib.request
from typing import List
from PIL import Image
import io
from app.config import settings
from app.utils.exceptions import VectorServiceError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating image embeddings"""

    def __init__(self):
        self.dimension = settings.vector_dimension
        self.use_real_model = False
        self.model = None
        self._initialized = False
        self.models_path = "/tmp/models"
        self.detection_model_path = f"{self.models_path}/detection.onnx"
        self.embedding_model_path = f"{self.models_path}/embedding.onnx"
        self.models_zip_path = "/tmp/models.zip"

        logger.info(
            "EmbeddingService created - models will be initialized on first use"
        )

    def _ensure_models_exist(self):
        """Ensure model files exist, download/extract if necessary"""
        try:
            # Check if models directory exists
            if not os.path.exists(self.models_path):
                os.makedirs(self.models_path, exist_ok=True)
                logger.info(f"Created models directory: {self.models_path}")

            # Check if model files exist
            if not os.path.exists(self.detection_model_path) or not os.path.exists(
                self.embedding_model_path
            ):
                logger.info("Model files not found, attempting to extract from zip...")

                # Check if zip file exists
                if os.path.exists(self.models_zip_path):
                    logger.info(
                        f"Found models zip at {self.models_zip_path}, extracting..."
                    )
                    self._extract_models_from_zip()
                else:
                    logger.warning(f"Models zip not found at {self.models_zip_path}")
                    return False

            # Verify both model files exist after extraction
            if os.path.exists(self.detection_model_path) and os.path.exists(
                self.embedding_model_path
            ):
                logger.info("Model files verified successfully")
                return True
            else:
                logger.error("Model files still missing after extraction attempt")
                return False

        except Exception as e:
            logger.error(f"Error ensuring models exist: {str(e)}")
            return False

    def _extract_models_from_zip(self):
        """Extract model files from zip archive"""
        try:
            with zipfile.ZipFile(self.models_zip_path, "r") as zip_ref:
                # Extract all files to /tmp/models
                zip_ref.extractall("/tmp/")
                logger.info("Successfully extracted models from zip")

                # List extracted files for debugging
                for root, dirs, files in os.walk(self.models_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        logger.info(f"Extracted file: {file_path}")

        except Exception as e:
            logger.error(f"Failed to extract models from zip: {str(e)}")
            raise

    def _initialize_model(self):
        """Initialize the face analysis model with lazy loading"""
        try:
            if self._initialized:
                return True

            logger.info("Initializing face analysis model...")

            # Ensure models exist first
            if not self._ensure_models_exist():
                logger.warning(
                    "Models not available, falling back to stub implementation"
                )
                self.use_real_model = False
                self._initialized = True
                return False

            # Check if model files exist
            if not os.path.exists(self.detection_model_path):
                raise FileNotFoundError(
                    f"Detection model not found at {self.detection_model_path}"
                )

            if not os.path.exists(self.embedding_model_path):
                raise FileNotFoundError(
                    f"Embedding model not found at {self.embedding_model_path}"
                )

            # Initialize your face analysis model here
            # This is where you would load your actual ONNX models
            # Example:
            # from app.core.src.face_analysis import FaceAnalysis
            # self.model = FaceAnalysis(...)

            logger.info("Face analysis model initialized successfully")
            self.use_real_model = True
            self._initialized = True
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize face analysis model: {str(e)}")
            logger.info("Falling back to stub implementation")
            self.use_real_model = False
            self._initialized = True
            return False

    async def generate_embedding(self, image_data: bytes) -> List[float]:
        """Generate embedding from image data with lazy initialization"""
        try:
            # Validate image first
            self._validate_image(image_data)

            # Initialize model on first use (lazy initialization)
            if not self._initialized:
                self._initialize_model()

            if self.use_real_model and self.model:
                # Use real model
                return await self._generate_real_embedding(image_data)
            else:
                # Use stub implementation
                return self._generate_stub_embedding()

        except Exception as e:
            raise VectorServiceError(f"Failed to generate embedding: {str(e)}")

    async def _generate_real_embedding(self, image_data: bytes) -> List[float]:
        """Generate embedding using real face analysis model"""
        try:
            # Implement your real model inference here
            # This would process the image through your ONNX models

            # Placeholder - replace with actual model inference
            logger.info("Generating embedding using real model")
            vector = np.random.random(self.dimension).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            return vector.tolist()

        except Exception as e:
            logger.error(f"Real model inference failed: {str(e)}")
            # Fallback to stub if real model fails
            return self._generate_stub_embedding()

    def _generate_stub_embedding(self) -> List[float]:
        """Generate random embedding for testing purposes"""
        logger.debug("Generating embedding using stub implementation")
        vector = np.random.random(self.dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()

    def _validate_image(self, image_data: bytes) -> None:
        """Validate image format and size"""
        try:
            # Check file size
            if len(image_data) > settings.max_file_size:
                raise VectorServiceError("Image file too large")

            # Validate image format
            image = Image.open(io.BytesIO(image_data))

            # Check if image format is supported
            if image.format.lower() not in ["jpeg", "jpg", "png"]:
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

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)

        except Exception as e:
            raise VectorServiceError(f"Failed to calculate similarity: {str(e)}")

    def get_model_status(self) -> dict:
        """Get current model status for health checks"""
        return {
            "initialized": self._initialized,
            "using_real_model": self.use_real_model,
            "detection_model_exists": os.path.exists(self.detection_model_path),
            "embedding_model_exists": os.path.exists(self.embedding_model_path),
            "models_zip_exists": os.path.exists(self.models_zip_path),
        }


# Create singleton instance but don't initialize models yet
embedding_service = EmbeddingService()
