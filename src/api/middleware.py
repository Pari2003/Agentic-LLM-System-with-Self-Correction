"""
CORS, timing, and structured logging middleware for the FastAPI application.

Provides:
1. Request timing — logs and returns X-Process-Time header
2. Structured request logging — structured log entry for every request
3. CORS — configurable cross-origin resource sharing
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings

logger = structlog.get_logger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """Attach all middleware to the FastAPI application instance.

    Args:
        app: The FastAPI application.
    """
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing and logging middleware
    @app.middleware("http")
    async def timing_and_logging_middleware(
        request: Request, call_next
    ) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        # Bind request context for structured logging
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        logger.info(
            "request_start",
            query_string=str(request.query_params) if request.query_params else None,
        )

        try:
            response = await call_next(request)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "request_error",
                error=str(e),
                elapsed_ms=round(elapsed_ms, 1),
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}ms"
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_complete",
            status_code=response.status_code,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return response
