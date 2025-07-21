from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
from app.utils.exceptions import AuthenticationError, create_http_exception

security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify the bearer token"""
    try:
        token = credentials.credentials

        # Simple token validation - in production, use proper JWT validation
        if token != settings.api_key:
            raise AuthenticationError("Invalid API key")

        return token
    except AuthenticationError as e:
        raise create_http_exception(e)
    except Exception:
        raise create_http_exception(AuthenticationError())