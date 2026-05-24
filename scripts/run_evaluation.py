"""
Evaluation Benchmark Runner.

Runs the full dual evaluation framework against the pipeline:
1. Retrieval Evaluation (free, no LLM) — Context Precision, Recall, MRR, NDCG, Hit Rate
2. LLM-as-Judge (requires LLM) — Faithfulness, Answer Relevancy, Completeness
3. Performance Metrics — Latency breakdown, correction rates, source usage

Usage:
    python -m scripts.run_evaluation --questions data/eval/test_questions.json
    python -m scripts.run_evaluation --questions data/eval/test_questions.json --skip-llm-judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import structlog

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.metrics_collector import MetricsCollector
from src.evaluation.retrieval_metrics import RetrievalEvaluator
from src.models.llm_client import OllamaClient
from src.models.schemas import EvalResult, QueryRequest

logger = structlog.get_logger(__name__)


async def run_retrieval_evaluation(
    questions: list[dict],
    k: int = 5,
) -> dict[str, float]:
    """Run retrieval-only evaluation (no LLM, no live pipeline needed).

    This evaluates retrieval quality using ground-truth chunk IDs.
    In a full benchmark, you would run the pipeline first to get
    retrieved chunk IDs, then compare against ground truth.

    For demonstration, this uses synthetic retrieval results.
    """
    evaluator = RetrievalEvaluator()

    print("\n" + "=" * 60)
    print("  STAGE 1: RETRIEVAL EVALUATION (No LLM)")
    print("=" * 60)

    # In a real scenario, you'd run the retrieval pipeline here.
    # For the benchmark, we demonstrate the evaluator with the test data.
    batch_data = []
    for q in questions:
        # Simulate retrieved chunks based on keyword matching
        # In production, this would come from actual pipeline retrieval
        retrieved = q.get("retrieved_chunk_ids", [])
        relevant = q.get("relevant_chunk_ids", [])

        if retrieved and relevant:
            batch_data.append({
                "retrieved_chunk_ids": retrieved,
                "relevant_chunk_ids": relevant,
            })

    if batch_data:
        avg_metrics = evaluator.evaluate_batch(batch_data, k=k)
        print(f"\n  Queries evaluated: {avg_metrics.get('num_queries', 0)}")
        for metric, value in avg_metrics.items():
            if metric != "num_queries":
                print(f"  {metric}: {value:.4f}")
        return avg_metrics
    else:
        print("  ⚠ No retrieval ground truth available in test questions.")
        print("  To run retrieval evaluation, add 'retrieved_chunk_ids' and")
        print("  'relevant_chunk_ids' fields to your test questions.")
        return {}


async def run_llm_judge_evaluation(
    questions: list[dict],
    llm_client: OllamaClient,
) -> dict[str, float]:
    """Run LLM-as-Judge evaluation on pre-generated answers.

    Requires answers to already be generated. Evaluates faithfulness,
    answer relevancy, and completeness.
    """
    judge = LLMJudge(llm_client)

    print("\n" + "=" * 60)
    print("  STAGE 2: LLM-AS-JUDGE EVALUATION")
    print("=" * 60)

    # Filter questions that have pre-generated answers
    evaluatable = [q for q in questions if q.get("predicted_answer")]

    if not evaluatable:
        print("  ⚠ No predicted answers found in test questions.")
        print("  Run the pipeline first to generate answers, then re-run evaluation.")
        return {}

    faithfulness_scores = []
    relevancy_scores = []
    completeness_scores = []

    for i, q in enumerate(evaluatable):
        query = q["query"]
        answer = q["predicted_answer"]
        contexts = q.get("contexts", [])

        print(f"\n  [{i+1}/{len(evaluatable)}] Evaluating: {query[:60]}...")

        result = await judge.evaluate(
            query=query,
            answer=answer,
            contexts=contexts,
        )

        faithfulness_scores.append(result.faithfulness)
        relevancy_scores.append(result.answer_relevancy)
        completeness_scores.append(result.completeness)

        print(f"    Faithfulness: {result.faithfulness:.3f}")
        print(f"    Relevancy:    {result.answer_relevancy:.3f}")
        print(f"    Completeness: {result.completeness:.3f}")

    n = len(evaluatable)
    avg_metrics = {
        "avg_faithfulness": round(sum(faithfulness_scores) / n, 4),
        "avg_relevancy": round(sum(relevancy_scores) / n, 4),
        "avg_completeness": round(sum(completeness_scores) / n, 4),
        "num_evaluated": n,
    }

    print(f"\n  ─── LLM Judge Summary ({n} queries) ───")
    print(f"  Avg Faithfulness:  {avg_metrics['avg_faithfulness']:.4f}")
    print(f"  Avg Relevancy:     {avg_metrics['avg_relevancy']:.4f}")
    print(f"  Avg Completeness:  {avg_metrics['avg_completeness']:.4f}")

    return avg_metrics


def display_final_report(
    retrieval_metrics: dict,
    generation_metrics: dict,
    performance_summary: dict,
) -> None:
    """Print a comprehensive final report."""
    print("\n")
    print("╔" + "═" * 60 + "╗")
    print("║" + "  DUAL EVALUATION BENCHMARK — FINAL REPORT".center(60) + "║")
    print("╠" + "═" * 60 + "╣")

    if retrieval_metrics:
        print("║" + "  RETRIEVAL METRICS (No LLM)".ljust(60) + "║")
        print("╠" + "─" * 60 + "╣")
        for key, val in retrieval_metrics.items():
            if key != "num_queries":
                line = f"  {key}: {val:.4f}"
                print("║" + line.ljust(60) + "║")
        print("╠" + "═" * 60 + "╣")

    if generation_metrics:
        print("║" + "  GENERATION METRICS (LLM-as-Judge)".ljust(60) + "║")
        print("╠" + "─" * 60 + "╣")
        for key, val in generation_metrics.items():
            if key != "num_evaluated":
                line = f"  {key}: {val:.4f}"
                print("║" + line.ljust(60) + "║")
        print("╠" + "═" * 60 + "╣")

    if performance_summary:
        print("║" + "  PERFORMANCE METRICS".ljust(60) + "║")
        print("╠" + "─" * 60 + "╣")
        for key, val in performance_summary.items():
            line = f"  {key}: {val}"
            print("║" + line.ljust(60) + "║")

    print("╚" + "═" * 60 + "╝")


async def main():
    parser = argparse.ArgumentParser(
        description="Run the Dual Evaluation Benchmark Suite"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/eval/test_questions.json",
        help="Path to the test questions JSON file",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM-as-Judge evaluation (only run retrieval metrics)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k value for retrieval evaluation metrics",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the evaluation results as JSON",
    )
    args = parser.parse_args()

    # Load test questions
    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"Error: Questions file not found at {questions_path}")
        sys.exit(1)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} test questions from {questions_path}")

    start_time = time.perf_counter()

    # Stage 1: Retrieval Evaluation
    retrieval_metrics = await run_retrieval_evaluation(questions, k=args.k)

    # Stage 2: LLM-as-Judge (optional)
    generation_metrics = {}
    if not args.skip_llm_judge:
        llm_client = OllamaClient()
        try:
            generation_metrics = await run_llm_judge_evaluation(questions, llm_client)
        finally:
            await llm_client.close()
    else:
        print("\n  ⏭ Skipping LLM-as-Judge evaluation (--skip-llm-judge)")

    elapsed = time.perf_counter() - start_time

    # Final report
    performance = {
        "total_evaluation_time_s": round(elapsed, 2),
        "total_questions": len(questions),
    }

    display_final_report(retrieval_metrics, generation_metrics, performance)

    # Save results
    if args.output:
        results = {
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "performance": performance,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
