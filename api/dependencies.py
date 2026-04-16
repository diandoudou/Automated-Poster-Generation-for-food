import logging
from typing import Optional

from api.config import settings
from fastapi import HTTPException, Header

logger = logging.getLogger(__name__)

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Verify API key if authentication is enabled.
    Add this as a dependency to protected endpoints.
    """
    # Skip if no API key is configured
    if not hasattr(settings, 'API_KEY') or not settings.API_KEY:
        return True

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required"
        )

    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return True

async def rate_limiter():
    """
    Simple rate limiting (implement with Redis for production)
    """
    # TODO: Implement proper rate limiting
    return True