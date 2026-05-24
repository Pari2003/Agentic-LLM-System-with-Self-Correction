"""
LLM-as-Judge Evaluator.

Uses Llama 3.2 to evaluate generation quality on three dimensions:
1. Faithfulness — Is every claim in the answer grounded in the source contexts?
2. Answer Relevancy — Does the answer actually address the user's question?
3. Completeness — Does the answer cover all aspects of the question?

This is the SECOND stage of the dual evaluation framework. Only run after
retrieval metrics are satisfactory (cost optimization — saves LLM calls).

Usage:
    from src.evaluation.llm_judge import LLMJudge

    judge = LLMJudge(llm_client)
    result = await judge.evaluate(
        query="What optimizer was used?",
        answer="Adam with lr=1e-4 [1].",
        contexts=["...passage 1...", "...passage 2..."],
    )
    print(result.faithfulness)  # 0.9
"""

from __future__ import annotations

import structlog

from src.models.llm_client import OllamaClient
from src.models.schemas import GenerationEvalResult

logger = structlog.get_logger(__name__)


class LLMJudge:
    """LLM-as-Judge evaluator for generation quality metrics.

    Uses the same Llama 3.2 instance as the pipeline agents to score
    faithfulness, answer relevancy, and completeness via structured JSON
    prompts. Temperature is set to 0.0 for deterministic scoring.
    """

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> GenerationEvalResult:
        """Run all three LLM-as-Judge evaluations.

        Args:
            query: The user's original question.
            answer: The generated answer to evaluate.
            contexts: List of source context texts used during generation.

        Returns:
            GenerationEvalResult with faithfulness, answer_relevancy, and completeness.
        """
        logger.info("llm_judge_start", query_len=len(query), answer_len=len(answer))

        faithfulness = await self._evaluate_faithfulness(answer, contexts)
        relevancy = await self._evaluate_relevancy(query, answer)
        completeness = await self._evaluate_completeness(query, answer, contexts)

        result = GenerationEvalResult(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            completeness=completeness,
        )

        logger.info(
            "llm_judge_complete",
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            completeness=result.completeness,
        )

        return result

    async def _evaluate_faithfulness(
        self, answer: str, contexts: list[str]
    ) -> float:
        """Score how well every claim in the answer is grounded in the sources.

        Returns a score between 0.0 (completely unfaithful) and 1.0 (fully grounded).
        """
        if not contexts:
            return 0.0

        context_str = "\n---\n".join(
            f"Source {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)
        )

        system_prompt = (
            "You are an impartial evaluation agent. Your task is to assess the "
            "faithfulness of an answer with respect to provided source contexts.\n"
            "Faithfulness means every factual claim in the answer must be directly "
            "supported by the source texts. Claims that are reasonable inferences "
            "but NOT explicitly stated in the sources should be penalized.\n\n"
            "Score the answer's faithfulness on a scale from 0.0 to 1.0:\n"
            "- 1.0: Every claim is directly supported by the sources.\n"
            "- 0.7-0.9: Most claims are supported, minor unsupported inferences.\n"
            "- 0.4-0.6: Mixed — some claims are supported, some are not.\n"
            "- 0.0-0.3: Most claims are fabricated or contradicted by sources.\n\n"
            "Output JSON: {\"score\": <float>, \"explanation\": \"<brief reason>\"}"
        )

        prompt = (
            f"Source Contexts:\n{context_str}\n\n"
            f"Answer to evaluate:\n{answer}\n\n"
            f"Faithfulness score:"
        )

        return await self._get_score(system_prompt, prompt, "faithfulness")

    async def _evaluate_relevancy(self, query: str, answer: str) -> float:
        """Score how well the answer addresses the user's question.

        Returns a score between 0.0 (completely off-topic) and 1.0 (perfectly relevant).
        """
        system_prompt = (
            "You are an impartial evaluation agent. Your task is to assess whether "
            "the answer directly and fully addresses the user's question.\n\n"
            "Score the answer relevancy on a scale from 0.0 to 1.0:\n"
            "- 1.0: The answer directly and precisely addresses the question.\n"
            "- 0.7-0.9: The answer mostly addresses the question with minor tangents.\n"
            "- 0.4-0.6: The answer partially addresses the question but misses key aspects.\n"
            "- 0.0-0.3: The answer is off-topic or does not address the question.\n\n"
            "Output JSON: {\"score\": <float>, \"explanation\": \"<brief reason>\"}"
        )

        prompt = (
            f"Question: {query}\n\n"
            f"Answer: {answer}\n\n"
            f"Relevancy score:"
        )

        return await self._get_score(system_prompt, prompt, "relevancy")

    async def _evaluate_completeness(
        self, query: str, answer: str, contexts: list[str]
    ) -> float:
        """Score how completely the answer covers all aspects of the question.

        Returns a score between 0.0 (nothing covered) and 1.0 (all aspects addressed).
        """
        if not contexts:
            return 0.0

        context_str = "\n---\n".join(
            f"Source {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)
        )

        system_prompt = (
            "You are an impartial evaluation agent. Your task is to assess whether "
            "the answer completely covers all aspects of the user's question, given "
            "the available source contexts.\n\n"
            "Score completeness on a scale from 0.0 to 1.0:\n"
            "- 1.0: The answer covers every relevant aspect found in the sources.\n"
            "- 0.7-0.9: Most aspects are covered, minor details missing.\n"
            "- 0.4-0.6: Several important aspects are missing from the answer.\n"
            "- 0.0-0.3: The answer is shallow or only covers a fraction of what is available.\n\n"
            "Output JSON: {\"score\": <float>, \"explanation\": \"<brief reason>\"}"
        )

        prompt = (
            f"Question: {query}\n\n"
            f"Source Contexts:\n{context_str}\n\n"
            f"Answer to evaluate:\n{answer}\n\n"
            f"Completeness score:"
        )

        return await self._get_score(system_prompt, prompt, "completeness")

    async def _get_score(
        self, system_prompt: str, prompt: str, metric_name: str
    ) -> float:
        """Send evaluation prompt to LLM and parse the numeric score from JSON response."""
        try:
            result = await self.llm_client.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
            )

            score = result.get("score", 0.0)
            # Clamp to [0.0, 1.0]
            score = max(0.0, min(1.0, float(score)))

            logger.debug(
                f"llm_judge_{metric_name}_scored",
                score=score,
                explanation=result.get("explanation", ""),
            )
            return round(score, 3)

        except Exception as e:
            logger.error(
                f"llm_judge_{metric_name}_error",
                error=str(e),
            )
            return 0.0
