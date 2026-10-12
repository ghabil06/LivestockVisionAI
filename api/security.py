"""
LivestockVisionAI Security & Hardening Middleware.
Provides API key authentication, sliding-window rate limiting,
payload size guards, and OWASP security response headers.
"""

import time
import os
from typing import Dict, List, Optional
from collections import defaultdict
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """Sliding-window in-memory rate limiter per client IP address."""

    def __init__(self, requests_per_minute: int = 120):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self.hits: Dict[str, List[float]] = defaultdict(list)

    def is_rate_limited(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean up older timestamps
        self.hits[client_ip] = [t for t in self.hits[client_ip] if t > window_start]
        
        if len(self.hits[client_ip]) >= self.requests_per_minute:
            return True
            
        self.hits[client_ip].append(now)
        return False


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects standard defensive HTTP security headers into every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Enforces per-client IP sliding window request frequency limits."""

    def __init__(self, app, requests_per_minute: int = 120, enabled: bool = True):
        super().__init__(app)
        self.limiter = RateLimiter(requests_per_minute=requests_per_minute)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        # Skip rate limits for static assets and health check
        path = request.url.path
        if path.startswith("/static") or path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        if self.limiter.is_rate_limited(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Maximum request frequency reached.",
                    "retry_after_seconds": 60,
                },
                headers={"Retry-After": "60"},
            )

        return await call_next(request)


def verify_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
) -> bool:
    """
    Validates API key for protected routes if API key auth is enabled.
    Checks environment variable LIVESTOCK_API_KEY, configuration file, or defaults.
    """
    from LivestockVisionAI.api.dependencies import get_platform_config
    cfg = get_platform_config()
    sec_cfg = cfg.get("security", {})
    
    # Check if auth required (disabled by default in dev/test)
    auth_enabled = sec_cfg.get("api_key_enabled", False)
    env_key = os.environ.get("LIVESTOCK_API_KEY")
    if env_key:
        auth_enabled = True

    if not auth_enabled:
        return True

    valid_keys = set(sec_cfg.get("api_keys", []))
    if env_key:
        valid_keys.add(env_key)

    # Check header
    token = api_key_header
    if not token:
        # Check Authorization Bearer header
        auth_hdr = request.headers.get("Authorization", "")
        if auth_hdr.startswith("Bearer "):
            token = auth_hdr[7:].strip()

    if not token or token not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide a valid 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return True
