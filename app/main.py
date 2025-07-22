from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import dedup
from app.utils.exceptions import DedupException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
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
        "INTERNAL_ERROR": 500
    }

    status_code = status_code_map.get(exc.code, 500)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": None
            }
        }
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "dedup-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)