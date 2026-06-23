"""
Retriever Agent.

Orchestrates the entire 3-stage retrieval pipeline:
1. Query Analysis (Intent, entities, sub-queries)
2. Hybrid Search (Vector + BM25 + Graph Boost via RRF)
3. Parent Expansion (Child -> Parent expansion + dedup)
4. LLM Reranking (Passage scoring on 1-10 scale)
"""


from __future__ import annotations

import uuid
import structlog

from src.storage.entity_graph import EntityGraph
from src.agents.base import BaseAgent
from src.agents.query_analyzer import QueryAnalyzer
from src.config import settings
from src.models.llm_client import OllamaClient
from src.models.schemas import QueryAnalysis, RankedContext
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.llm_reranker import LLMReranker
from src.retrieval.parent_expander import ParentExpander

logger = structlog.get_logger(__name__)


class Retriever(BaseAgent):
    """Orchestrator agent for the 3-stage retrieval pipeline."""

    def __init__(
        self,
        llm_client: OllamaClient,
        query_analyzer: QueryAnalyzer,
        hybrid_search: HybridSearch,
        parent_expander: ParentExpander,
        llm_reranker: LLMReranker,
        entity_graph: EntityGraph,
    ):
        super().__init__(llm_client)
        self.entity_graph = entity_graph
        self.query_analyzer = query_analyzer
        self.hybrid_search = hybrid_search
        self.parent_expander = parent_expander
        self.llm_reranker = llm_reranker

    async def retrieve(
        self,
        query: str,
        session_id: str,
        max_sources: int = settings.llm_rerank_top_k,
    ) -> tuple[list[RankedContext], QueryAnalysis, dict[str, float]]:
        """Run the full 3-stage retrieval pipeline, tracking latency.

        Args:
            query: The user query.
            session_id: The active session identifier.
            max_sources: Number of final context items to return.

        Returns:
            A tuple containing:
            - List of RankedContext objects
            - QueryAnalysis object
            - Dictionary of latency metrics in milliseconds
        """
        self.logger.info("retrieval_pipeline_start", query=query, session_id=session_id)
        pipeline_timer = self.start_timer()
        latencies: dict[str, float] = {}

        # ─── Step 1: Query Analysis ───
        step_timer = self.start_timer()
        query_analysis = await self.query_analyzer.analyze(query)
        latencies["query_analysis_ms"] = self.stop_timer_and_log("step_query_analysis", step_timer)

        # ─── Step 2: Embed Query ───
        step_timer = self.start_timer()
        try:
            # Embed primary search query from rewritten queries, default to original
            primary_query = (
                query_analysis.search_queries[0] if query_analysis.search_queries else query
            )
            query_vector = await self.llm_client.embed_single(primary_query)
        except Exception as e:
            self.logger.error("query_embedding_failed", error=str(e))
            # Create a zero vector if embedding fails to avoid crashing
            query_vector = [0.0] * settings.embedding_dim
        latencies["query_embedding_ms"] = self.stop_timer_and_log(
            "step_query_embedding", step_timer
        )

        # ─── Step 3: Stage 1 Hybrid Search ───
        step_timer = self.start_timer()
        child_results = await self.hybrid_search.search(
            query=query,
            query_vector=query_vector,
            session_id=session_id,
            extracted_entities=query_analysis.extracted_entities,
            top_k=settings.hybrid_search_top_k,
        )
        latencies["hybrid_search_ms"] = self.stop_timer_and_log(
            "step_hybrid_search", step_timer, num_children=len(child_results)
        )
        # # Save Question -> Chunk retrieval graph
        # try:
        #     question_id = str(uuid.uuid4())

        #     self.entity_graph.save_query_chunk_relationships(
        #     question=query,
        #     question_id=question_id,
        #     session_id=session_id,
        #     retrieved_chunks=child_results,
        #     )

        # except Exception as e:
        #     self.logger.warning(
        #     "query_chunk_graph_save_failed",
        #     error=str(e),
        #     )

        # ─── Step 4: Stage 2 Parent Expansion ───
        step_timer = self.start_timer()
        parent_candidates = self.parent_expander.expand(
            results=child_results,
            top_n=settings.parent_expansion_top_k,
        )
        latencies["parent_expansion_ms"] = self.stop_timer_and_log(
            "step_parent_expansion", step_timer, num_parents=len(parent_candidates)
        )

        # ─── Step 5: Stage 3 LLM Reranking ───
        step_timer = self.start_timer()
        final_contexts = await self.llm_reranker.rerank(
            query=query,
            contexts=parent_candidates,
            top_k=max_sources,
        )
        latencies["llm_rerank_ms"] = self.stop_timer_and_log(
            "step_llm_rerank", step_timer, final_contexts_count=len(final_contexts)
        )

        try:
            question_id = str(uuid.uuid4())

            self.entity_graph.save_final_context_graph(
            question=query,
            question_id=question_id,
            session_id=session_id,
            final_contexts=final_contexts,
        )

        except Exception as e:
            self.logger.warning(
            "final_context_graph_failed",
            error=str(e),
        )

        total_ms = self.stop_timer_and_log(
            "retrieval_pipeline",
            pipeline_timer,
            total_contexts=len(final_contexts),
        )
        latencies["total_ms"] = total_ms

        return final_contexts, query_analysis, latencies
