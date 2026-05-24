"""
Evaluation package.

Exposes hallucination detection, confidence scoring, and evaluation metrics.
"""

from src.evaluation.confidence_scorer import ConfidenceScorer
from src.evaluation.hallucination_detector import HallucinationDetector

__all__ = [
    "HallucinationDetector",
    "ConfidenceScorer",
]
