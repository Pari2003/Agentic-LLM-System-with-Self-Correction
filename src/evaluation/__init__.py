"""
Evaluation package.

Exposes hallucination detection, confidence scoring, retrieval metrics,
LLM-as-Judge generation evaluation, and performance metrics collection.
"""

from src.evaluation.confidence_scorer import ConfidenceScorer
from src.evaluation.hallucination_detector import HallucinationDetector
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.metrics_collector import MetricsCollector
from src.evaluation.retrieval_metrics import RetrievalEvaluator

__all__ = [
    "HallucinationDetector",
    "ConfidenceScorer",
    "RetrievalEvaluator",
    "LLMJudge",
    "MetricsCollector",
]
