from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.routers import dedup
from app.utils.exceptions import DedupException
from app.services.embedding import embedding_service
from app.services.redis_service import redis_service
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events"""
    # Startup
    logger.info("Starting up Deduplication API...")

    try:
        # Test Redis connection
        redis_healthy = await redis_service.health_check()
        if redis_healthy:
            logger.info("✓ Redis connection successful")
        else:
            logger.warning("⚠ Redis connection failed")

        # Download models if needed
        logger.info("Checking for models...")
        try:
            await embedding_service.download_models_if_needed()
            await embedding_service.initialize_if_models_available()
        except Exception as e:
            logger.warning(f"Model setup failed: {str(e)}")

        # Check embedding service status
        model_status = embedding_service.get_model_status()
        logger.info(f"Embedding service status: {model_status}")

        logger.info("✓ Application startup completed successfully")

    except Exception as e:
        logger.error(f"✗ Startup failed: {str(e)}")
        # Don't raise exception - let app start with degraded functionality

    yield

    # Shutdown
    logger.info("Shutting down Deduplication API...")
    logger.info("✓ Application shutdown completed")


# Create FastAPI app with lifespan events
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dedup.router)


@app.exception_handler(DedupException)
async def dedup_exception_handler(request: Request, exc: DedupException):
    """Global exception handler for DedupException"""
    status_code_map = {
        "AUTHENTICATION_FAILED": 401,
        "CUSTOMER_NOT_FOUND": 404,
        "INVALID_REQUEST": 400,
        "CUSTOMER_EXISTS": 409,
        "VECTOR_SERVICE_ERROR": 500,
        "INTERNAL_ERROR": 500,
    }

    status_code = status_code_map.get(exc.code, 500)

    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": exc.code, "message": exc.message, "details": None}},
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health():
    """Basic health check endpoint"""
    try:
        # Check Redis health
        redis_healthy = await redis_service.health_check()

        # Check embedding service status
        model_status = embedding_service.get_model_status()

        overall_health = redis_healthy and model_status.get("initialized", False)

        return {
            "status": "healthy" if overall_health else "degraded",
            "service": "dedup-api",
            "components": {
                "redis": "healthy" if redis_healthy else "unhealthy",
                "embedding": "healthy"
                if model_status.get("initialized", False)
                else "initializing",
                "model_status": model_status,
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "service": "dedup-api", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
