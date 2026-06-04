"""
Critic Agent.

Responsible for the multi-layer verification check on the generated answer.
Decomposes answers into atomic claims, verifies each in parallel, and scores them.
"""

from __future__ import annotations

import asyncio

from src.agents.base import BaseAgent
from src.evaluation.confidence_scorer import ConfidenceScorer
from src.evaluation.hallucination_detector import HallucinationDetector
from src.models.llm_client import OllamaClient
from src.models.schemas import CritiqueResult, RankedContext


class Critic(BaseAgent):
    """Critic Agent orchestrating claim extraction, parallel verification, and scoring."""

    def __init__(self, llm_client: OllamaClient):
        super().__init__(llm_client)
        self.detector = HallucinationDetector(llm_client)
        self.scorer = ConfidenceScorer()

    async def critique(
        self,
        query: str,
        answer: str,
        contexts: list[RankedContext],
        confidence_threshold: float = 0.70,
    ) -> CritiqueResult:
        """Analyze the generated answer for factual accuracy against the retrieved contexts.

        Args:
            query: The user query.
            answer: The generated answer to verify.
            contexts: List of RankedContext parent chunks used for generation.
            confidence_threshold: Confidence threshold below which refinement is required.

        Returns:
            A CritiqueResult object containing detailed claim verifications.
        """
        start_time = self.start_timer()
        self.logger.info("critique_start", answer_len=len(answer), contexts_count=len(contexts))

        # 1. Extract claims from response
        claims = await self.detector.extract_claims(answer)
        if not claims:
            # No factual claims found in response (e.g. "I don't know")
            empty_result = self.scorer.score_response([], threshold=confidence_threshold)
            self.stop_timer_and_log("critique", start_time, total_claims=0, hallucination_rate=0.0)
            return empty_result

        # 2. Extract texts and IDs from contexts
        context_texts = [ctx.parent_chunk.text for ctx in contexts]
        context_ids = [ctx.parent_chunk.id for ctx in contexts]

        # 3. Verify all claims concurrently
        tasks = [self.detector.verify_claim(claim, context_texts, context_ids) for claim in claims]
        verifications = await asyncio.gather(*tasks)

        # 4. Score overall response
        critique_result = self.scorer.score_response(verifications, threshold=confidence_threshold)

        self.stop_timer_and_log(
            "critique",
            start_time,
            total_claims=critique_result.total_claims,
            hallucinated_count=len(critique_result.hallucinated_claims),
            overall_confidence=critique_result.overall_confidence,
            needs_refinement=critique_result.needs_refinement,
        )

        return critique_result
