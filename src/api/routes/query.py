"""
Query routes.

Endpoints:
- POST /sessions/{session_id}/query — Run a question through the full agentic QA pipeline
"""

from __future__ import annotations

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_orchestrator, get_doc_store
from src.models.schemas import (
    Citation,
    ClaimVerification,
    CritiqueResult,
    PipelineMetrics,
    QueryAnalysis,
    QueryRequest,
    QueryResponse,
)
from src.pipeline.orchestrator import Orchestrator
from src.storage.document_store import DocumentStore

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Query"])


# ─── Request / Response Models ────────────────────────────────────────────

class QueryRequestBody(BaseModel):
    """Request body for submitting a query."""
    question: str = Field(..., min_length=3, max_length=2000, description="The question to ask")
    max_sources: int = Field(default=5, ge=1, le=20, description="Maximum number of source contexts to retrieve")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence to accept without correction")
    enable_self_correction: bool = Field(default=True, description="Enable the agentic self-correction loop")
    max_correction_iterations: int = Field(default=2, ge=0, le=5, description="Maximum self-correction iterations")


class ClaimDetail(BaseModel):
    """Simplified claim verification detail for the API response."""
    claim_text: str
    is_hallucination: bool
    confidence: float
    explanation: Optional[str] = None


class QueryResponseBody(BaseModel):
    """Response model for query results."""
    answer: str
    citations: list[Citation] = []
    confidence_score: float
    correction_iterations: int = 0
    total_claims: int = 0
    hallucination_rate: float = 0.0
    hallucinated_claims: list[ClaimDetail] = []
    query_analysis: QueryAnalysis
    metrics: PipelineMetrics
    session_id: str


# ─── Routes ───────────────────────────────────────────────────────────────

@router.post(
    "/sessions/{session_id}/query",
    response_model=QueryResponseBody,
    summary="Query Documents",
    description=(
        "Submit a question to be answered using the full agentic RAG pipeline. "
        "The pipeline runs: Query Analysis → Hybrid Search → Parent Expansion → "
        "LLM Reranking → Generation → Critique → Self-Correction (if needed)."
    ),
)
async def query_documents(
    session_id: str,
    body: QueryRequestBody,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    doc_store: DocumentStore = Depends(get_doc_store),
):
    """Run the full agentic QA pipeline for a user question."""
    # Validate session
    session = doc_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if not session.is_active:
        raise HTTPException(status_code=410, detail=f"Session '{session_id}' has expired")

    logger.info(
        "query_start",
        session_id=session_id,
        question=body.question[:100],
    )

    try:
        # Build internal query request
        request = QueryRequest(
            question=body.question,
            session_id=session_id,
            max_sources=body.max_sources,
            confidence_threshold=body.confidence_threshold,
            enable_self_correction=body.enable_self_correction,
            max_correction_iterations=body.max_correction_iterations,
        )

        # Run the full pipeline
        response: QueryResponse = await orchestrator.run(request)

        # Build simplified hallucination details
        hallucinated_details = []
        for v in response.claim_report.hallucinated_claims:
            hallucinated_details.append(ClaimDetail(
                claim_text=v.claim.text,
                is_hallucination=v.is_hallucination,
                confidence=v.overall_confidence,
                explanation=v.explanation,
            ))

        # Increment session query count
        doc_store.increment_session_query_count(session_id)

        return QueryResponseBody(
            answer=response.answer,
            citations=response.citations,
            confidence_score=response.confidence_score,
            correction_iterations=response.correction_iterations,
            total_claims=response.claim_report.total_claims,
            hallucination_rate=response.claim_report.hallucination_rate,
            hallucinated_claims=hallucinated_details,
            query_analysis=response.query_analysis,
            metrics=response.metrics,
            session_id=session_id,
        )

    except Exception as e:
        logger.error("query_error", error=str(e), session_id=session_id)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
