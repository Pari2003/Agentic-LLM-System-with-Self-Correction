"""
Stage 3 LLM Reranking.

Uses Ollama (Llama 3.2 3B) to score the relevance of parent passages
on a 1-10 scale relative to the query. Executes queries concurrently.
"""

from __future__ import annotations

import asyncio
import re
import time

import structlog

from src.config import settings
from src.models.llm_client import OllamaClient
from src.models.schemas import RankedContext

logger = structlog.get_logger(__name__)


def extract_score(response_text: str) -> float:
    """Extract a 1-10 integer score from the LLM response text."""
    clean_text = response_text.strip()

    # Try to find exactly 1 to 10 bounded by word boundaries
    matches = re.findall(r"\b([1-9]|10)\b", clean_text)
    if matches:
        return float(matches[0])

    # Fallback to any numeric decimal sequence
    matches_float = re.findall(r"\d+\.?\d*", clean_text)
    if matches_float:
        val = float(matches_float[0])
        return min(max(val, 1.0), 10.0)

    return 1.0  # Safe default if no score found


class LLMReranker:
    """Grades and reranks parent chunks using Llama 3.2 relevance scoring."""

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    async def _score_single_passage(
        self,
        query: str,
        context: RankedContext,
    ) -> float:
        """Call Ollama to grade the relevance of a single passage."""
        prompt = f"""You are an expert scientific paper reviewer and search grader.
Evaluate the relevance of the following passage to the search query.
Rate the passage on a scale of 1 to 10, where:
- 1: Completely irrelevant (the passage has nothing to do with the query)
- 5: Moderately relevant (contains matching terms but does not directly answer)
- 10: Extremely relevant (directly and completely answers the query)

Search Query: "{query}"

Passage:
---
{context.parent_chunk.text}
---

Provide ONLY a single integer score between 1 and 10. Do not write any explanations, preamble, or markdown. Just return the digit.
Score:"""

        start = time.perf_counter()
        try:
            # Low temperature for deterministic scoring
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=5,
            )
            score = extract_score(response)
            elapsed = (time.perf_counter() - start) * 1000

            logger.debug(
                "llm_rerank_score_passage",
                parent_id=context.parent_chunk.id,
                raw_response=response.strip(),
                parsed_score=score,
                elapsed_ms=round(elapsed, 1),
            )
            return score
        except Exception as e:
            logger.error(
                "llm_rerank_score_failed",
                parent_id=context.parent_chunk.id,
                error=str(e),
            )
            return 1.0

    async def rerank(
        self,
        query: str,
        contexts: list[RankedContext],
        top_k: int = settings.llm_rerank_top_k,
    ) -> list[RankedContext]:
        """Grade and rerank all candidate contexts concurrently.

        Args:
            query: The search query.
            contexts: The candidate parent chunks.
            top_k: The final number of contexts to retain.

        Returns:
            The sorted and truncated list of RankedContext items.
        """
        if not contexts:
            return []

        logger.info("llm_reranking_start", query=query, num_contexts=len(contexts), top_k=top_k)
        start_time = time.perf_counter()

        # Score all passages concurrently
        tasks = [self._score_single_passage(query, ctx) for ctx in contexts]
        scores = await asyncio.gather(*tasks)

        # Apply scores
        for idx, score in enumerate(scores):
            contexts[idx].llm_relevance_score = score

        # Sort: Primary key is LLM relevance score descending, secondary key is RRF score descending
        contexts.sort(key=lambda x: (x.llm_relevance_score or 0.0, x.rrf_score), reverse=True)
        top_contexts = contexts[:top_k]

        # Re-assign final rank
        for rank_idx, ctx in enumerate(top_contexts):
            ctx.final_rank = rank_idx + 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "llm_reranking_complete",
            elapsed_ms=round(elapsed_ms, 1),
            retained_count=len(top_contexts),
        )
        return top_contexts
