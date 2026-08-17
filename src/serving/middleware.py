"""
Middleware
==========
Cross-cutting concerns applied to every request:
    1. Request ID injection   → unique UUID per request for tracing
    2. Latency logging        → log every request with timing
    3. SLA monitoring         → warn/error if latency exceeds targets
"""

import time
import uuid
from loguru import logger
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_settings


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique request_id into every request.
    Makes distributed tracing and log correlation possible.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())

        # Inject into request state — accessible in route handlers
        request.state.request_id = request_id

        # Add to response headers for client-side tracing
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class LatencyMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Measures and logs end-to-end request latency.
    Fires warnings/errors if SLA targets are breached.
    """

    def __init__(self, app, settings=None):
        super().__init__(app)
        self.settings = settings or get_settings()

    async def dispatch(self, request: Request, call_next) -> Response:
        start    = time.perf_counter()
        response = await call_next(request)
        elapsed  = (time.perf_counter() - start) * 1000    # ms

        # Add latency to response headers
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.2f}"

        # Route-aware logging
        path   = request.url.path
        method = request.method
        status = response.status_code

        # SLA checks — only for recommendation endpoints
        if "/recommend" in path:
            if elapsed > self.settings.latency_error_ms:
                logger.error(
                    f"SLA BREACH | {method} {path} | "
                    f"status={status} | latency={elapsed:.1f}ms | "
                    f"threshold={self.settings.latency_error_ms}ms"
                )
            elif elapsed > self.settings.latency_warn_ms:
                logger.warning(
                    f"SLA WARNING | {method} {path} | "
                    f"status={status} | latency={elapsed:.1f}ms | "
                    f"threshold={self.settings.latency_warn_ms}ms"
                )
            else:
                logger.info(
                    f"{method} {path} | "
                    f"status={status} | latency={elapsed:.1f}ms"
                )
        else:
            logger.debug(
                f"{method} {path} | status={status} | {elapsed:.1f}ms"
            )

        return response