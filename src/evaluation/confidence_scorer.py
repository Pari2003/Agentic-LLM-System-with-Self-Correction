"""
Confidence Scorer.

Aggregates individual claim verification results into a global critique report
and computes overall confidence, hallucination rates, and refinement triggers.
"""

from __future__ import annotations

from typing import Optional

import structlog

from src.models.schemas import ClaimVerification, CritiqueResult

logger = structlog.get_logger(__name__)


class ConfidenceScorer:
    """Aggregates claim verifications to score overall answer quality and trigger refinement."""

    def __init__(self, default_threshold: float = 0.70):
        self.default_threshold = default_threshold

    def score_response(
        self,
        verifications: list[ClaimVerification],
        threshold: Optional[float] = None,
    ) -> CritiqueResult:
        """Aggregate verification list into a CritiqueResult.

        Args:
            verifications: List of individual ClaimVerification objects.
            threshold: Confidence threshold below which refinement is triggered.
                       Defaults to default_threshold (0.70).

        Returns:
            A CritiqueResult object.
        """
        threshold = threshold if threshold is not None else self.default_threshold
        total_claims = len(verifications)

        if total_claims == 0:
            logger.info("scoring_empty_claims")
            return CritiqueResult(
                claims=[],
                overall_confidence=1.0,
                hallucinated_claims=[],
                supported_claims=[],
                total_claims=0,
                hallucination_rate=0.0,
                needs_refinement=False,
            )

        hallucinated = []
        supported = []
        confidence_sum = 0.0

        for v in verifications:
            confidence_sum += v.overall_confidence
            if v.is_hallucination:
                hallucinated.append(v)
            else:
                supported.append(v)

        overall_conf = confidence_sum / total_claims
        hallucination_rate = len(hallucinated) / total_claims

        # Response needs refinement if the aggregate score is below threshold
        # OR if there are any explicitly hallucinated claims.
        needs_refinement = (overall_conf < threshold) or (len(hallucinated) > 0)

        logger.info(
            "response_scored",
            total_claims=total_claims,
            overall_confidence=round(overall_conf, 3),
            hallucinated_count=len(hallucinated),
            hallucination_rate=round(hallucination_rate, 3),
            needs_refinement=needs_refinement,
        )

        return CritiqueResult(
            claims=verifications,
            overall_confidence=round(overall_conf, 3),
            hallucinated_claims=hallucinated,
            supported_claims=supported,
            total_claims=total_claims,
            hallucination_rate=round(hallucination_rate, 3),
            needs_refinement=needs_refinement,
        )
