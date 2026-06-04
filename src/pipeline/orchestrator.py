"""
Orchestrator.

Orchestrates the entire agentic QA pipeline:
1. Stage 1-3 Retrieval (Retriever agent)
2. Response Generation (Generator agent)
3. Factual Accuracy Critique (Critic agent)
4. Iterative Self-Correction Loop (Refiner agent)
"""

from __future__ import annotations

import time

import structlog

from src.agents.critic import Critic
from src.agents.generator import Generator
from src.agents.refiner import Refiner
from src.agents.retriever import Retriever
from src.models.schemas import PipelineMetrics, QueryRequest, QueryResponse


class Orchestrator:
    """Pipeline Orchestrator implementing the full agentic QA loop with self-correction."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        critic: Critic,
        refiner: Refiner,
    ):
        self.retriever = retriever
        self.generator = generator
        self.critic = critic
        self.refiner = refiner
        self.logger = structlog.get_logger(self.__class__.__name__)

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Execute the full agentic QA pipeline for a user request.

        Args:
            request: The QueryRequest containing question, session_id, and limits.

        Returns:
            A QueryResponse object containing answer, citations, report, and metrics.
        """
        start_time = time.perf_counter()
        self.logger.info(
            "pipeline_start",
            question=request.question,
            session_id=request.session_id,
            enable_self_correction=request.enable_self_correction,
        )

        # ─── 1. Retrieval Stage (Analyzer -> Hybrid Search -> Expansion -> Reranking) ───
        contexts, query_analysis, retriever_latencies = await self.retriever.retrieve(
            query=request.question,
            session_id=request.session_id,
            max_sources=request.max_sources,
        )

        # ─── 2. Answer Generation Stage ───
        start_gen = time.perf_counter()
        answer, citations = await self.generator.generate(
            query=request.question,
            contexts=contexts,
            session_id=request.session_id,
        )
        generation_ms = (time.perf_counter() - start_gen) * 1000

        # ─── 3. Critique Stage ───
        start_crit = time.perf_counter()
        critique_result = await self.critic.critique(
            query=request.question,
            answer=answer,
            contexts=contexts,
            confidence_threshold=request.confidence_threshold,
        )
        critique_ms = (time.perf_counter() - start_crit) * 1000

        # ─── 4. Self-Correction Loop ───
        correction_iterations = 0
        refinement_ms_accumulated = 0.0
        critique_ms_accumulated = critique_ms

        if request.enable_self_correction and critique_result.needs_refinement:
            max_iters = request.max_correction_iterations
            self.logger.info(
                "self_correction_triggered",
                overall_confidence=critique_result.overall_confidence,
                hallucinated_claims_count=len(critique_result.hallucinated_claims),
                max_iterations=max_iters,
            )

            while critique_result.needs_refinement and correction_iterations < max_iters:
                correction_iterations += 1
                self.logger.info(
                    "correction_iteration_start",
                    iteration=correction_iterations,
                    max_iterations=max_iters,
                )

                # Refinement step
                start_ref = time.perf_counter()
                answer, citations = await self.refiner.refine(
                    query=request.question,
                    original_answer=answer,
                    critique_result=critique_result,
                    contexts=contexts,
                )
                refinement_ms_accumulated += (time.perf_counter() - start_ref) * 1000

                # Re-critique step
                start_crit = time.perf_counter()
                critique_result = await self.critic.critique(
                    query=request.question,
                    answer=answer,
                    contexts=contexts,
                    confidence_threshold=request.confidence_threshold,
                )
                critique_ms_accumulated += (time.perf_counter() - start_crit) * 1000

                self.logger.info(
                    "correction_iteration_complete",
                    iteration=correction_iterations,
                    new_confidence=critique_result.overall_confidence,
                    needs_refinement=critique_result.needs_refinement,
                )

        # ─── 5. Metrics Aggregation ───
        total_ms = (time.perf_counter() - start_time) * 1000

        metrics = PipelineMetrics(
            query_analysis_ms=retriever_latencies.get("query_analysis_ms", 0.0),
            metadata_filter_ms=retriever_latencies.get("metadata_filter_ms", 0.0),
            hybrid_search_ms=retriever_latencies.get("hybrid_search_ms", 0.0),
            parent_expansion_ms=retriever_latencies.get("parent_expansion_ms", 0.0),
            llm_rerank_ms=retriever_latencies.get("llm_rerank_ms", 0.0),
            generation_ms=round(generation_ms, 1),
            critique_ms=round(critique_ms_accumulated, 1),
            refinement_ms=round(refinement_ms_accumulated, 1),
            total_ms=round(total_ms, 1),
            correction_iterations=correction_iterations,
            retrieval_sources_used=len(contexts),
        )

        self.logger.info(
            "pipeline_complete",
            session_id=request.session_id,
            final_confidence=critique_result.overall_confidence,
            correction_iterations=correction_iterations,
            total_time_ms=round(total_ms, 1),
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            confidence_score=critique_result.overall_confidence,
            claim_report=critique_result,
            query_analysis=query_analysis,
            metrics=metrics,
            correction_iterations=correction_iterations,
            session_id=request.session_id,
        )
