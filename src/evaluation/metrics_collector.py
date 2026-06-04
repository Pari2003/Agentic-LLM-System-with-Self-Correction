"""
Metrics Collector.

Aggregates latency, token usage, and correction rates across pipeline runs.
Provides per-query and aggregate statistics for performance analysis and
bottleneck detection.

Usage:
    from src.evaluation.metrics_collector import MetricsCollector

    collector = MetricsCollector()
    collector.record(pipeline_metrics)
    summary = collector.summary()
    print(summary["avg_total_ms"])
"""

from __future__ import annotations

import statistics

import structlog

from src.models.schemas import PipelineMetrics

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Collects and aggregates pipeline performance metrics across queries.

    Tracks per-stage latencies, correction rates, and source usage to identify
    bottlenecks and measure system performance over time.
    """

    def __init__(self):
        self._records: list[PipelineMetrics] = []

    @property
    def count(self) -> int:
        """Number of recorded pipeline runs."""
        return len(self._records)

    def record(self, metrics: PipelineMetrics) -> None:
        """Record a single pipeline run's metrics.

        Args:
            metrics: PipelineMetrics from a completed query.
        """
        self._records.append(metrics)
        logger.debug(
            "metrics_recorded",
            total_ms=metrics.total_ms,
            correction_iterations=metrics.correction_iterations,
            count=len(self._records),
        )

    def record_batch(self, metrics_list: list[PipelineMetrics]) -> None:
        """Record metrics from multiple pipeline runs.

        Args:
            metrics_list: List of PipelineMetrics objects.
        """
        for m in metrics_list:
            self.record(m)

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._records.clear()
        logger.info("metrics_collector_reset")

    def summary(self) -> dict[str, float]:
        """Compute aggregate statistics across all recorded runs.

        Returns:
            Dict with averaged, median, min, max, and p95 metrics.
            Returns empty dict if no records exist.
        """
        if not self._records:
            return {}

        n = len(self._records)

        # Collect per-stage latencies
        total_ms = [r.total_ms for r in self._records]
        query_analysis_ms = [r.query_analysis_ms for r in self._records]
        hybrid_search_ms = [r.hybrid_search_ms for r in self._records]
        parent_expansion_ms = [r.parent_expansion_ms for r in self._records]
        llm_rerank_ms = [r.llm_rerank_ms for r in self._records]
        generation_ms = [r.generation_ms for r in self._records]
        critique_ms = [r.critique_ms for r in self._records]
        refinement_ms = [r.refinement_ms for r in self._records]
        correction_iterations = [r.correction_iterations for r in self._records]
        sources_used = [r.retrieval_sources_used for r in self._records]

        summary = {
            # ─── Total Pipeline ───
            "num_queries": n,
            "avg_total_ms": round(statistics.mean(total_ms), 1),
            "median_total_ms": round(statistics.median(total_ms), 1),
            "min_total_ms": round(min(total_ms), 1),
            "max_total_ms": round(max(total_ms), 1),
            "p95_total_ms": round(self._percentile(total_ms, 95), 1),
            # ─── Per-Stage Averages ───
            "avg_query_analysis_ms": round(statistics.mean(query_analysis_ms), 1),
            "avg_hybrid_search_ms": round(statistics.mean(hybrid_search_ms), 1),
            "avg_parent_expansion_ms": round(statistics.mean(parent_expansion_ms), 1),
            "avg_llm_rerank_ms": round(statistics.mean(llm_rerank_ms), 1),
            "avg_generation_ms": round(statistics.mean(generation_ms), 1),
            "avg_critique_ms": round(statistics.mean(critique_ms), 1),
            "avg_refinement_ms": round(statistics.mean(refinement_ms), 1),
            # ─── Self-Correction ───
            "avg_correction_iterations": round(statistics.mean(correction_iterations), 2),
            "correction_trigger_rate": round(sum(1 for c in correction_iterations if c > 0) / n, 3),
            "max_correction_iterations": max(correction_iterations),
            # ─── Source Usage ───
            "avg_sources_used": round(statistics.mean(sources_used), 1),
        }

        logger.info("metrics_summary_computed", **summary)
        return summary

    def per_stage_breakdown(self) -> dict[str, dict[str, float]]:
        """Compute detailed per-stage latency breakdown.

        Returns a dict where each key is a stage name and the value is a dict
        with avg, median, min, max, and p95 latencies.
        """
        if not self._records:
            return {}

        stages = {
            "query_analysis": [r.query_analysis_ms for r in self._records],
            "metadata_filter": [r.metadata_filter_ms for r in self._records],
            "hybrid_search": [r.hybrid_search_ms for r in self._records],
            "parent_expansion": [r.parent_expansion_ms for r in self._records],
            "llm_rerank": [r.llm_rerank_ms for r in self._records],
            "generation": [r.generation_ms for r in self._records],
            "critique": [r.critique_ms for r in self._records],
            "refinement": [r.refinement_ms for r in self._records],
        }

        breakdown = {}
        for stage_name, values in stages.items():
            breakdown[stage_name] = {
                "avg_ms": round(statistics.mean(values), 1),
                "median_ms": round(statistics.median(values), 1),
                "min_ms": round(min(values), 1),
                "max_ms": round(max(values), 1),
                "p95_ms": round(self._percentile(values, 95), 1),
            }

        return breakdown

    def to_table(self) -> str:
        """Format the summary as a human-readable table string.

        Returns:
            Formatted multi-line string suitable for printing or logging.
        """
        summary = self.summary()
        if not summary:
            return "No metrics recorded."

        lines = [
            "╔══════════════════════════════════════════════════════════╗",
            "║              PIPELINE PERFORMANCE SUMMARY               ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Queries Evaluated:          {summary['num_queries']:>8}                  ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  LATENCY (ms)          Avg      Median    P95     Max   ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Total Pipeline     {summary['avg_total_ms']:>8.1f}   {summary['median_total_ms']:>8.1f}   {summary['p95_total_ms']:>6.1f}  {summary['max_total_ms']:>6.1f} ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  PER-STAGE AVERAGES                                     ║",
            f"║    Query Analysis:     {summary['avg_query_analysis_ms']:>8.1f} ms                       ║",
            f"║    Hybrid Search:      {summary['avg_hybrid_search_ms']:>8.1f} ms                       ║",
            f"║    Parent Expansion:   {summary['avg_parent_expansion_ms']:>8.1f} ms                       ║",
            f"║    LLM Rerank:         {summary['avg_llm_rerank_ms']:>8.1f} ms                       ║",
            f"║    Generation:         {summary['avg_generation_ms']:>8.1f} ms                       ║",
            f"║    Critique:           {summary['avg_critique_ms']:>8.1f} ms                       ║",
            f"║    Refinement:         {summary['avg_refinement_ms']:>8.1f} ms                       ║",
            "╠══════════════════════════════════════════════════════════╣",
            "║  SELF-CORRECTION                                        ║",
            f"║    Trigger Rate:            {summary['correction_trigger_rate']:>5.1%}                       ║",
            f"║    Avg Iterations:          {summary['avg_correction_iterations']:>5.2f}                       ║",
            f"║    Max Iterations:          {summary['max_correction_iterations']:>5}                       ║",
            "╠══════════════════════════════════════════════════════════╣",
            f"║  Avg Sources Used:          {summary['avg_sources_used']:>5.1f}                       ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    @staticmethod
    def _percentile(values: list[float], pct: int) -> float:
        """Compute the Nth percentile of a list of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = (pct / 100) * (len(sorted_vals) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_vals) - 1)
        weight = idx - lower
        return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight
