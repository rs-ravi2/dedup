from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
from app.utils.exceptions import AuthenticationError, create_http_exception
from typing import Optional

security = HTTPBearer(auto_error=False)  # Don't auto-error to handle custom logic


async def verify_token(request: Request,
                       credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """Verify the bearer token or direct API key"""
    try:
        token = None

        # Try to get token from Bearer format first
        if credentials:
            token = credentials.credentials
        else:
            # Fallback: get token directly from Authorization header
            auth_header = request.headers.get("Authorization")
            if auth_header:
                # Remove 'Bearer ' prefix if present, otherwise use as-is
                token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else auth_header

        if not token:
            raise AuthenticationError("Missing API key")

        # Simple token validation - in production, use proper JWT validation
        if token != settings.api_key:
            raise AuthenticationError("Invalid API key")

        return token
    except AuthenticationError as e:
        raise create_http_exception(e)
    except Exception:
        raise create_http_exception(AuthenticationError())