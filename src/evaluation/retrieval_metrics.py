"""
Retrieval Evaluation Metrics.

Computes retrieval quality metrics WITHOUT any LLM calls — these are fast and free.
Metrics include Context Precision, Context Recall, MRR@k, NDCG@k, and Hit Rate@k.

Usage:
    from src.evaluation.retrieval_metrics import RetrievalEvaluator

    evaluator = RetrievalEvaluator()
    result = evaluator.evaluate(
        retrieved_chunk_ids=["c1", "c2", "c3", "c4", "c5"],
        relevant_chunk_ids=["c2", "c5", "c9"],
        k=5,
    )
    print(result.mrr_at_k)  # 0.5
"""

from __future__ import annotations

import math
from typing import Optional

import structlog

from src.models.schemas import RetrievalEvalResult

logger = structlog.get_logger(__name__)


class RetrievalEvaluator:
    """Computes retrieval quality metrics against ground-truth relevant chunk IDs.

    All metrics are computed deterministically with no LLM calls,
    making them suitable for rapid iteration during retrieval tuning.
    """

    def evaluate(
        self,
        retrieved_chunk_ids: list[str],
        relevant_chunk_ids: list[str],
        k: Optional[int] = None,
    ) -> RetrievalEvalResult:
        """Evaluate retrieval quality for a single query.

        Args:
            retrieved_chunk_ids: Ordered list of chunk IDs returned by retrieval
                                (best match first).
            relevant_chunk_ids: Ground-truth set of chunk IDs that are relevant
                                to the query.
            k: Number of top results to evaluate. Defaults to length of
               retrieved_chunk_ids.

        Returns:
            RetrievalEvalResult with all computed metrics.
        """
        if not retrieved_chunk_ids or not relevant_chunk_ids:
            logger.warning(
                "retrieval_eval_empty_input",
                retrieved_count=len(retrieved_chunk_ids),
                relevant_count=len(relevant_chunk_ids),
            )
            return RetrievalEvalResult(k=k or 5)

        k = k or len(retrieved_chunk_ids)
        top_k = retrieved_chunk_ids[:k]
        relevant_set = set(relevant_chunk_ids)

        context_precision = self._context_precision(top_k, relevant_set)
        context_recall = self._context_recall(top_k, relevant_set)
        mrr = self._mrr_at_k(top_k, relevant_set)
        ndcg = self._ndcg_at_k(top_k, relevant_set)
        hit_rate = self._hit_rate_at_k(top_k, relevant_set)

        result = RetrievalEvalResult(
            context_precision=round(context_precision, 4),
            context_recall=round(context_recall, 4),
            mrr_at_k=round(mrr, 4),
            ndcg_at_k=round(ndcg, 4),
            hit_rate_at_k=round(hit_rate, 4),
            k=k,
        )

        logger.info(
            "retrieval_eval_complete",
            k=k,
            context_precision=result.context_precision,
            context_recall=result.context_recall,
            mrr_at_k=result.mrr_at_k,
            ndcg_at_k=result.ndcg_at_k,
            hit_rate_at_k=result.hit_rate_at_k,
        )

        return result

    def evaluate_batch(
        self,
        queries: list[dict],
        k: Optional[int] = None,
    ) -> dict[str, float]:
        """Evaluate retrieval quality across multiple queries and return averaged metrics.

        Args:
            queries: List of dicts, each with keys:
                     - 'retrieved_chunk_ids': list[str]
                     - 'relevant_chunk_ids': list[str]
            k: Number of top results per query.

        Returns:
            Dict with averaged metric names and values.
        """
        if not queries:
            return {}

        results = []
        for q in queries:
            result = self.evaluate(
                retrieved_chunk_ids=q["retrieved_chunk_ids"],
                relevant_chunk_ids=q["relevant_chunk_ids"],
                k=k,
            )
            results.append(result)

        n = len(results)
        avg = {
            "avg_context_precision": round(sum(r.context_precision for r in results) / n, 4),
            "avg_context_recall": round(sum(r.context_recall for r in results) / n, 4),
            "avg_mrr_at_k": round(sum(r.mrr_at_k for r in results) / n, 4),
            "avg_ndcg_at_k": round(sum(r.ndcg_at_k for r in results) / n, 4),
            "avg_hit_rate_at_k": round(sum(r.hit_rate_at_k for r in results) / n, 4),
            "num_queries": n,
        }

        logger.info("retrieval_eval_batch_complete", **avg)
        return avg

    # ─── Individual Metrics ───────────────────────────────────────────────

    @staticmethod
    def _context_precision(top_k: list[str], relevant: set[str]) -> float:
        """Proportion of retrieved items in top-k that are relevant.

        Also known as Precision@k.
        Formula: |relevant ∩ retrieved@k| / k
        """
        if not top_k:
            return 0.0
        hits = sum(1 for cid in top_k if cid in relevant)
        return hits / len(top_k)

    @staticmethod
    def _context_recall(top_k: list[str], relevant: set[str]) -> float:
        """Proportion of all relevant items that appear in the retrieved top-k.

        Formula: |relevant ∩ retrieved@k| / |relevant|
        """
        if not relevant:
            return 0.0
        hits = sum(1 for cid in top_k if cid in relevant)
        return hits / len(relevant)

    @staticmethod
    def _mrr_at_k(top_k: list[str], relevant: set[str]) -> float:
        """Mean Reciprocal Rank — reciprocal of the rank of the first relevant result.

        Formula: 1 / rank_of_first_relevant_hit
        Returns 0.0 if no relevant result is found in top-k.
        """
        for rank, cid in enumerate(top_k, start=1):
            if cid in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _ndcg_at_k(top_k: list[str], relevant: set[str]) -> float:
        """Normalized Discounted Cumulative Gain at k.

        Uses binary relevance (1 if relevant, 0 if not).
        Formula: DCG@k / IDCG@k
        """
        if not top_k or not relevant:
            return 0.0

        # DCG: sum of 1 / log2(rank + 1) for each relevant item
        dcg = 0.0
        for rank, cid in enumerate(top_k, start=1):
            if cid in relevant:
                dcg += 1.0 / math.log2(rank + 1)

        # IDCG: best possible DCG (all relevant items ranked first)
        ideal_hits = min(len(relevant), len(top_k))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _hit_rate_at_k(top_k: list[str], relevant: set[str]) -> float:
        """Binary indicator: 1.0 if ANY relevant item appears in top-k, else 0.0."""
        for cid in top_k:
            if cid in relevant:
                return 1.0
        return 0.0
