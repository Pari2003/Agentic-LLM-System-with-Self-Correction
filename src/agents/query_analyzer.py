"""
Query Analyzer Agent.

Decomposes and analyzes incoming questions using Llama 3.2 3B.
Classifies complexity, extracts key entities for graph boost lookups,
and generates search queries optimized for vector and keyword engines.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.agents.base import BaseAgent
from src.models.schemas import QueryAnalysis, QueryComplexity

logger = structlog.get_logger(__name__)


class QueryAnalyzer(BaseAgent):
    """Agent that understands and decomposes queries prior to retrieval."""

    async def analyze(self, query: str) -> QueryAnalysis:
        """Analyze the query and decompose it into search targets.

        Args:
            query: The raw input question from the user.

        Returns:
            Structured QueryAnalysis containing classifications and sub-queries.
        """
        self.logger.info("query_analysis_start", query=query)
        timer = self.start_timer()

        system_prompt = (
            "You are an expert Query Analyzer for an AI/ML research paper Q&A system.\n"
            "Analyze the user query and return a structured JSON response specifying:\n"
            "- complexity: 'simple' (single-hop lookup), 'moderate' (multi-fact/comparison), or 'complex' (multi-hop reasoning)\n"
            "- sub_queries: List of simpler sub-questions required to answer the query (empty if simple)\n"
            "- extracted_entities: List of proper names, methods, datasets, or metrics (e.g. 'Transformer', 'BLEU', 'SGD', 'WMT 2014')\n"
            "- search_queries: List of rewritten keyword/semantic search phrases optimized for document retrieval\n"
            "- intent: A short statement summarizing the core question intent.\n\n"
            "Rules:\n"
            "1. Output ONLY valid JSON containing EXACTLY the keys: "
            "['complexity', 'sub_queries', 'extracted_entities', 'search_queries', 'intent']\n"
            "2. Do not write any preamble, explanation, or postscript."
        )

        prompt = f"Query: \"{query}\"\n\nJSON Output:"

        analysis_dict: dict[str, Any] = {}
        try:
            analysis_dict = await self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,  # Deterministic analysis
            )
        except Exception as e:
            self.logger.error("query_analyzer_call_failed", error=str(e))

        # Graceful fallback mapping
        complexity_val = analysis_dict.get("complexity", "simple").lower()
        if complexity_val not in ["simple", "moderate", "complex"]:
            complexity = QueryComplexity.SIMPLE
        else:
            complexity = QueryComplexity(complexity_val)

        sub_queries = analysis_dict.get("sub_queries")
        if not isinstance(sub_queries, list):
            sub_queries = []
            
        extracted_entities = analysis_dict.get("extracted_entities")
        if not isinstance(extracted_entities, list):
            extracted_entities = []
            
        search_queries = analysis_dict.get("search_queries")
        if not isinstance(search_queries, list) or not search_queries:
            # Fall back to searching the original query
            search_queries = [query]
            
        intent = analysis_dict.get("intent", "Information retrieval query")

        analysis = QueryAnalysis(
            original_query=query,
            complexity=complexity,
            sub_queries=sub_queries,
            extracted_entities=extracted_entities,
            search_queries=search_queries,
            intent=intent,
        )

        self.stop_timer_and_log(
            "query_analysis",
            timer,
            complexity=complexity.value,
            num_sub_queries=len(sub_queries),
            num_entities=len(extracted_entities),
        )
        return analysis
