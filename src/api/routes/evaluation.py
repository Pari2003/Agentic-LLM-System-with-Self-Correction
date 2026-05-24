"""
Evaluation routes.

Endpoints:
- POST /evaluation/retrieval — Run retrieval metrics on provided data
- POST /evaluation/generation — Run LLM-as-Judge on a generated answer
"""

from __future__ import annotations

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_llm_client
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.retrieval_metrics import RetrievalEvaluator
from src.models.llm_client import OllamaClient
from src.models.schemas import GenerationEvalResult, RetrievalEvalResult

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


# ─── Request / Response Models ────────────────────────────────────────────

class RetrievalEvalRequest(BaseModel):
    """Request body for retrieval evaluation."""
    retrieved_chunk_ids: list[str] = Field(..., description="Ordered list of retrieved chunk IDs")
    relevant_chunk_ids: list[str] = Field(..., description="Ground-truth relevant chunk IDs")
    k: int = Field(default=5, ge=1, le=50, description="Top-k cutoff for metrics")


class GenerationEvalRequest(BaseModel):
    """Request body for generation evaluation."""
    query: str = Field(..., description="The original user question")
    answer: str = Field(..., description="The generated answer to evaluate")
    contexts: list[str] = Field(default=[], description="Source context texts used for generation")


# ─── Routes ───────────────────────────────────────────────────────────────

@router.post(
    "/retrieval",
    response_model=RetrievalEvalResult,
    summary="Evaluate Retrieval Quality",
    description=(
        "Compute retrieval metrics (Context Precision, Recall, MRR@k, NDCG@k, Hit Rate@k) "
        "against ground-truth relevant chunk IDs. No LLM calls — fast and free."
    ),
)
async def evaluate_retrieval(body: RetrievalEvalRequest):
    """Compute retrieval evaluation metrics."""
    evaluator = RetrievalEvaluator()

    result = evaluator.evaluate(
        retrieved_chunk_ids=body.retrieved_chunk_ids,
        relevant_chunk_ids=body.relevant_chunk_ids,
        k=body.k,
    )

    return result


@router.post(
    "/generation",
    response_model=GenerationEvalResult,
    summary="Evaluate Generation Quality",
    description=(
        "Use LLM-as-Judge to evaluate generation quality on Faithfulness, "
        "Answer Relevancy, and Completeness. Requires LLM calls."
    ),
)
async def evaluate_generation(
    body: GenerationEvalRequest,
    llm_client: OllamaClient = Depends(get_llm_client),
):
    """Run LLM-as-Judge evaluation on a generated answer."""
    if not body.contexts:
        raise HTTPException(
            status_code=400,
            detail="At least one context passage is required for generation evaluation.",
        )

    judge = LLMJudge(llm_client)

    result = await judge.evaluate(
        query=body.query,
        answer=body.answer,
        contexts=body.contexts,
    )

    return result
