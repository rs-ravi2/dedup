import numpy as np
from typing import List
import cv2
import io
import os
import shutil
import sys
import logging
import time
from app.config import settings
from app.utils.exceptions import VectorServiceError
from app.core.src.face_analysis import FaceAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_max_area_dict(predictions):
    """Get the face detection with maximum area from multiple detections."""
    if not predictions:
        return []
    
    max_area = 0
    max_prediction = None
    
    for pred in predictions:
        bbox = pred.bbox
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > max_area:
            max_area = area
            max_prediction = pred
    
    return [max_prediction] if max_prediction else []


class EmbeddingService:
    """Service for generating image embeddings using face analysis"""

    def __init__(self):
        self.dimension = settings.vector_dimension
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the face analysis model"""
        try:
            detection_model = "/tmp/models/detection.onnx"
            embedding_model = "/tmp/models/embedding.onnx"
            
            if not os.path.exists(detection_model):
                logger.error(f"Detection Model Does not exist at path: {str(detection_model)}")
            if not os.path.exists(embedding_model):
                logger.error(f"embedding Model Does not exist at path: {str(embedding_model)}")
            
            # Initialize Face Analysis Model
            self.model = FaceAnalysis(
                allowed_modules=['detection', 'recognition'], 
                providers=['CPUExecutionProvider']
            )
            self.model.prepare(ctx_id=0)
            logger.info("Face analysis model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face analysis model: {str(e)}")
            raise VectorServiceError(f"Model initialization failed: {str(e)}")

    def generate_embedding(self, image_data: bytes) -> List[float]:
        """Generate embedding from image data using face analysis"""
        try:
            t1 = time.perf_counter()
            
            # Validate image
            self._validate_image(image_data)
            
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise VectorServiceError("Failed to decode image")
            
            # Get face predictions
            predictions = self.model.get(image)
            
            if len(predictions) == 0:
                raise VectorServiceError("No face detected in image")
            
            # If multiple faces detected, get the one with maximum area
            if len(predictions) > 1:
                predictions = get_max_area_dict(predictions)
            
            # Extract embedding
            embedding = predictions[0].embedding
            
            # Ensure embedding is normalized and convert to list
            embedding = embedding / np.linalg.norm(embedding)
            
            logger.info(f"Embedding generation completed in {time.perf_counter() - t1:.4f} seconds")
            
            return embedding.tolist()

        except VectorServiceError:
            raise
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise VectorServiceError(f"Failed to generate embedding: {str(e)}")

    def _validate_image(self, image_data: bytes) -> None:
        """Validate image format and size"""
        try:
            # Check file size
            if len(image_data) > settings.max_file_size:
                raise VectorServiceError("Image file too large")
            
            # Basic validation - try to decode with OpenCV
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise VectorServiceError("Invalid image format or corrupted image")
            
            # Check minimum dimensions
            height, width = image.shape[:2]
            if width < 50 or height < 50:
                raise VectorServiceError("Image too small - minimum 50x50 pixels required")
                
        except VectorServiceError:
            raise
        except Exception as e:
            raise VectorServiceError(f"Image validation failed: {str(e)}")


embedding_service = EmbeddingService()
