"""
Health check routes.

Provides system health endpoints:
- /health — overall system health + Ollama model readiness
- /health/ready — lightweight readiness probe for k8s/docker
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_llm_client
from src.models.llm_client import OllamaClient

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    summary="System Health Check",
    description="Reports Ollama server status, available models, and readiness of text/embedding/vision models.",
)
async def health_check(llm_client: OllamaClient = Depends(get_llm_client)):
    """Full health check including Ollama model availability."""
    health = await llm_client.health_check()
    return health


@router.get(
    "/ready",
    summary="Readiness Probe",
    description="Lightweight readiness probe. Returns 200 if the API server is running.",
)
async def readiness():
    """Simple readiness probe — confirms the API is accepting requests."""
    return {"status": "ready"}
