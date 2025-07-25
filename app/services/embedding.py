import numpy as np
import os
import logging
import zipfile
import cv2
from typing import List, Optional
import io
from app.config import settings
from app.utils.exceptions import VectorServiceError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating image embeddings using existing core face analysis"""

    def __init__(self):
        self.dimension = settings.vector_dimension
        self.use_real_model = False
        self.face_analysis = None
        self._initialized = False

        # Model paths (from constants)
        self.models_path = "/tmp/models"
        self.detection_model_path = f"{self.models_path}/detection.onnx"
        self.embedding_model_path = f"{self.models_path}/embedding.onnx"
        self.models_zip_path = "/tmp/models.zip"

        logger.info("EmbeddingService initialized - models will be loaded on demand")

    def _check_models_exist(self) -> bool:
        """Check if required model files exist"""
        detection_exists = os.path.exists(self.detection_model_path)
        embedding_exists = os.path.exists(self.embedding_model_path)

        logger.debug(
            f"Model files - Detection: {detection_exists}, Embedding: {embedding_exists}"
        )
        return detection_exists and embedding_exists

    async def download_models_if_needed(self) -> bool:
        """Download and extract models if they don't exist"""
        try:
            # Check if models already exist
            if self._check_models_exist():
                logger.info("Model files already exist")
                return True

            # Check if zip exists and extract
            if os.path.exists(self.models_zip_path):
                logger.info("Models zip found, extracting...")
                return self._extract_models_from_zip()

            logger.warning(
                "Models not found. Please ensure models.zip exists at /tmp/models.zip"
            )
            return False

        except Exception as e:
            logger.error(f"Error in model setup: {str(e)}")
            return False

    def _extract_models_from_zip(self) -> bool:
        """Extract model files from zip archive"""
        try:
            os.makedirs(self.models_path, exist_ok=True)

            with zipfile.ZipFile(self.models_zip_path, "r") as zip_ref:
                zip_ref.extractall("/tmp/")
                logger.info("Models extracted successfully")

                # Log extracted files
                for root, dirs, files in os.walk(self.models_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        logger.info(f"Extracted: {file_path} ({size_mb:.1f}MB)")

            return self._check_models_exist()

        except Exception as e:
            logger.error(f"Failed to extract models: {str(e)}")
            return False

    async def initialize_if_models_available(self) -> bool:
        """Initialize models if they're available - called at startup"""
        try:
            if self._initialized:
                return self.use_real_model

            logger.info("Attempting to initialize face analysis models...")

            # Check if models exist
            if not self._check_models_exist():
                logger.info("Model files not found, staying in stub mode")
                self._initialized = True
                return False

            # Try to initialize the face analysis model
            return self._initialize_face_analysis()

        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            self._initialized = True
            return False

    def _initialize_face_analysis(self) -> bool:
        """Initialize the face analysis model using existing core implementation"""
        try:
            # Import from the existing core module
            from app.core.src.face_analysis import FaceAnalysis

            logger.info(
                "Initializing FaceAnalysis with existing core implementation..."
            )

            # Initialize with allowed modules for face detection and recognition

            self.face_analysis = FaceAnalysis(allowed_modules=['detection', 'recognition'],
                                              providers=['CPUExecutionProvider'])
            self.face_analysis.prepare(ctx_id=0)

            logger.info("✅ Face analysis model initialized successfully")
            logger.info(f"Available models: {list(self.face_analysis.models.keys())}")

            self.use_real_model = True
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize face analysis model: {str(e)}")
            logger.info("Falling back to stub implementation")
            self.use_real_model = False
            self._initialized = True
            return False

    async def generate_embedding(self, image_data: bytes) -> List[float]:
        """Generate embedding from image data"""
        try:
            # Validate image
            self._validate_image(image_data)

            # Initialize model on first use if not already done
            if not self._initialized:
                await self.initialize_if_models_available()

            # Use real model or fallback to stub
            if self.use_real_model and self.face_analysis:
                return await self._generate_real_embedding(image_data)
            else:
                return self._generate_stub_embedding()

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise VectorServiceError(f"Failed to generate embedding: {str(e)}")

    async def _generate_real_embedding(self, image_data: bytes) -> List[float]:
        """Generate embedding using real face analysis model"""
        try:
            # Convert bytes to numpy array using cv2
            nparr = np.frombuffer(image_data, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_array is None:
                logger.error("Failed to decode image with cv2")
                return self._generate_stub_embedding()

            # Convert BGR to RGB (cv2 loads as BGR, but face analysis expects RGB)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            logger.debug(f"Processing image: {img_array.shape}")

            # Use face analysis model to get faces
            # The get() method returns Face objects with embeddings
            faces = self.face_analysis.get(img_array, max_num=1)  # Get max 1 face

            if not faces:
                logger.warning("No faces detected in image, using stub embedding")
                return self._generate_stub_embedding()

            # Get the first detected face
            face = faces[0]

            # Extract embedding from the Face object
            # The Face object should have an 'embedding' attribute after processing
            if hasattr(face, "embedding") and face.embedding is not None:
                embedding = face.embedding

                # Normalize the embedding
                embedding = embedding / np.linalg.norm(embedding)

                logger.debug(f"Generated real embedding with shape: {embedding.shape}")
                return embedding.tolist()
            else:
                logger.warning("Face detected but no embedding generated, using stub")
                return self._generate_stub_embedding()

        except Exception as e:
            logger.error(f"Real model inference failed: {str(e)}")
            # Fallback to stub implementation
            logger.info("Falling back to stub implementation")
            return self._generate_stub_embedding()

    def _generate_stub_embedding(self) -> List[float]:
        """Generate random embedding for testing/fallback"""
        logger.debug("Generating stub embedding")

        # Generate a random but reproducible embedding for consistency in testing
        np.random.seed(42)  # Fixed seed for reproducible results
        vector = np.random.random(self.dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        return vector.tolist()

    def _validate_image(self, image_data: bytes) -> None:
        """Validate image format and size using cv2"""
        try:
            # Check file size
            if len(image_data) > settings.max_file_size:
                raise VectorServiceError("Image file too large")

            # Try to decode image with cv2
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise VectorServiceError(
                    "Invalid image format - cannot decode with cv2"
                )

            # Check minimum size
            height, width = image.shape[:2]
            if width < 50 or height < 50:
                raise VectorServiceError("Image too small")

            logger.debug(f"Image validation passed: {width}x{height}")

        except Exception as e:
            if isinstance(e, VectorServiceError):
                raise
            raise VectorServiceError(f"Invalid image format: {str(e)}")

    def calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)

            # Handle dimension mismatch
            if len(v1) != len(v2):
                logger.warning(f"Vector dimension mismatch: {len(v1)} vs {len(v2)}")
                return 0.0

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)
            # Ensure similarity is in valid range
            return float(np.clip(similarity, -1.0, 1.0))

        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            raise VectorServiceError(f"Failed to calculate similarity: {str(e)}")

    def get_model_status(self) -> dict:
        """Get current model status for health checks"""
        model_info = {}

        # Get available models if initialized
        if self.face_analysis and hasattr(self.face_analysis, "models"):
            model_info = {
                "available_models": list(self.face_analysis.models.keys()),
                "model_count": len(self.face_analysis.models),
            }

        return {
            "initialized": self._initialized,
            "using_real_model": self.use_real_model,
            "detection_model_exists": os.path.exists(self.detection_model_path),
            "embedding_model_exists": os.path.exists(self.embedding_model_path),
            "models_zip_exists": os.path.exists(self.models_zip_path),
            "models_path": self.models_path,
            "face_analysis_info": model_info,
            "file_sizes": {
                "detection": self._get_file_size(self.detection_model_path),
                "embedding": self._get_file_size(self.embedding_model_path),
            },
        }

    def _get_file_size(self, file_path: str) -> Optional[str]:
        """Get human-readable file size"""
        try:
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                size_mb = size_bytes / (1024 * 1024)
                return f"{size_mb:.1f}MB"
            return None
        except Exception:
            return None


# Global instance
embedding_service = EmbeddingService()
