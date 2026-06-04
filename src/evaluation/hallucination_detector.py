"""
Hallucination Detector.

Implements the multi-layer hallucination detection algorithm:
Layer 1: Cosine embedding similarity between atomic claims and retrieved parent chunks.
Layer 2: NLI (Natural Language Inference) entailment check via Llama 3.2.
Layer 3: Named entity and numerical overlap check.
"""

from __future__ import annotations

import re

import numpy as np
import structlog

from src.models.llm_client import OllamaClient
from src.models.schemas import Claim, ClaimVerification, EntailmentResult

logger = structlog.get_logger(__name__)


class HallucinationDetector:
    """Detects hallucinations in LLM-generated answers using a 3-layer verification process."""

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    async def extract_claims(self, response_text: str) -> list[Claim]:
        """Extract atomic factual claims from the response text using the LLM.

        Args:
            response_text: The full response text to decompose.

        Returns:
            A list of Claim objects.
        """
        logger.info("claim_extraction_start", text_len=len(response_text))

        system_prompt = (
            "You are a linguistic analysis agent. Your task is to break down a passage of text "
            "into individual, self-contained atomic factual claims.\n"
            "An atomic claim is a single statement that contains exactly one fact, and can be verified "
            "independently. Do not include opinions, meta-commentary, or questions. "
            "For each claim, you must also provide the exact sentence from the original passage "
            "that the claim was extracted from.\n"
            "You must output JSON in the following format:\n"
            "{\n"
            '  "claims": [\n'
            '    {"text": "Claim text", "source_sentence": "Original sentence"}\n'
            "  ]\n"
            "}"
        )

        prompt = f"Decompose the following passage into atomic factual claims:\n\n{response_text}"

        result_json = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )

        claims_data = result_json.get("claims", [])
        claims = []
        for idx, item in enumerate(claims_data):
            text = item.get("text", "").strip()
            source_sentence = item.get("source_sentence", "").strip()
            if text and source_sentence:
                claims.append(
                    Claim(
                        text=text,
                        source_sentence=source_sentence,
                    )
                )

        logger.info("claim_extraction_complete", extracted_claims_count=len(claims))
        return claims

    async def verify_claim(
        self,
        claim: Claim,
        context_texts: list[str],
        context_ids: list[str],
    ) -> ClaimVerification:
        """Verify an individual claim against the retrieved contexts using the 3-layer checks.

        Args:
            claim: The atomic Claim to verify.
            context_texts: List of retrieved parent chunk texts.
            context_ids: List of matching parent chunk IDs.

        Returns:
            A ClaimVerification object containing layers scores.
        """
        if not context_texts:
            return ClaimVerification(
                claim=claim,
                is_hallucination=True,
                explanation="No source contexts available for verification.",
            )

        # ─── Layer 1: Embedding Similarity ───
        claim_emb = await self.llm_client.embed_single(claim.text)
        context_embs = await self.llm_client.embed(context_texts)

        similarities = []
        for c_emb in context_embs:
            # Cosine similarity
            dot = np.dot(claim_emb, c_emb)
            norm_a = np.linalg.norm(claim_emb)
            norm_b = np.linalg.norm(c_emb)
            sim = float(dot / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
            similarities.append(sim)

        best_idx = int(np.argmax(similarities))
        best_sim = similarities[best_idx]
        best_text = context_texts[best_idx]
        best_chunk_id = context_ids[best_idx]

        # Initialize defaults
        entailment_res = EntailmentResult.NEUTRAL
        entailment_score = 0.5
        explanation = None

        # ─── Layer 2: NLI Check ───
        # Optimization: Only run expensive NLI if embedding similarity is not extremely high (e.g. < 0.82)
        # or if it's very low (e.g. > 0.40) to save calls on completely unrelated claims.
        if best_sim >= 0.82:
            # Highly grounded by vector representation
            entailment_res = EntailmentResult.SUPPORTS
            entailment_score = 1.0
            explanation = "High embedding similarity indicates alignment with source."
        elif best_sim < 0.35:
            # Completely ungrounded
            entailment_res = EntailmentResult.CONTRADICTS
            entailment_score = 0.0
            explanation = "Low embedding similarity indicates no semantic overlap with source."
        else:
            # Run LLM-as-NLI to resolve ambiguity
            entailment_res, entailment_score, explanation = await self._run_nli_llm(
                claim.text, best_text
            )

        # ─── Layer 3: Keyword/Numerical Overlap ───
        kw_score, matched_kws, missing_kws = self._run_keyword_check(claim.text, best_text)

        # ─── Combined Confidence Scoring ───
        # Formula: 0.4 * embedding_similarity + 0.4 * entailment_score + 0.2 * keyword_overlap_score
        overall_confidence = (0.4 * best_sim) + (0.4 * entailment_score) + (0.2 * kw_score)

        # We consider a claim hallucinated if overall confidence < 0.70 OR it contradicts the source.
        is_hallucination = (
            overall_confidence < 0.70 or entailment_res == EntailmentResult.CONTRADICTS
        )

        return ClaimVerification(
            claim=claim,
            embedding_similarity=best_sim,
            best_matching_chunk_id=best_chunk_id,
            best_matching_text=best_text,
            entailment_result=entailment_res,
            entailment_score=entailment_score,
            keyword_overlap_score=kw_score,
            matched_keywords=matched_kws,
            missing_keywords=missing_kws,
            overall_confidence=round(overall_confidence, 3),
            is_hallucination=is_hallucination,
            explanation=explanation or f"Combined confidence score: {overall_confidence:.2f}.",
        )

    async def _run_nli_llm(self, claim: str, source: str) -> tuple[EntailmentResult, float, str]:
        """Run Llama 3.2 to classify entailment (supports, contradicts, neutral)."""
        system_prompt = (
            "You are an NLI (Natural Language Inference) grading agent.\n"
            "Your task is to judge whether the provided source text supports or contradicts a specific claim.\n"
            "Select one of the following classes:\n"
            "- supports: The source text directly entails or provides clear evidence for the claim.\n"
            "- contradicts: The source text directly contradicts, falsifies, or negates the claim.\n"
            "- neutral: The source text does not contain enough information to verify or refute the claim.\n\n"
            "Provide your judgment in JSON format:\n"
            "{\n"
            '  "judgment": "supports" | "contradicts" | "neutral",\n'
            '  "explanation": "Brief explanation of your decision"\n'
            "}"
        )

        prompt = f"Source text:\n{source}\n\nClaim to verify:\n{claim}\n\nJudgment:"

        res_json = await self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )

        judgment = res_json.get("judgment", "neutral").lower().strip()
        explanation = res_json.get("explanation", "NLI evaluation completed.")

        if judgment == "supports":
            return EntailmentResult.SUPPORTS, 1.0, explanation
        elif judgment == "contradicts":
            return EntailmentResult.CONTRADICTS, 0.0, explanation
        else:
            return EntailmentResult.NEUTRAL, 0.5, explanation

    def _run_keyword_check(self, claim: str, source: str) -> tuple[float, list[str], list[str]]:
        """Extract numbers and entities/proper nouns from claim and check overlap in source."""
        # 1. Extract numbers (integers, floats, percentages)
        numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", claim)

        # 2. Extract capitalized proper nouns / candidate entities (excluding short words like I, etc.)
        # Pattern: matches capitalized words of length >= 2
        proper_nouns = re.findall(r"\b[A-Z][a-zA-Z0-9-]+\b", claim)

        # Filter stop words from proper nouns (e.g. "The", "A", "In", "On", "Of", "And")
        stop_proper = {
            "The",
            "A",
            "An",
            "In",
            "On",
            "Of",
            "And",
            "To",
            "For",
            "With",
            "By",
            "At",
            "From",
        }
        proper_nouns = [w for w in proper_nouns if w not in stop_proper]

        # Merge extracted keywords
        keywords = list(set(numbers + proper_nouns))

        if not keywords:
            # If no keywords or numbers exist, default to 1.0 (so we don't penalize simple statements)
            return 1.0, [], []

        matched = []
        missing = []
        source_lower = source.lower()

        for kw in keywords:
            # Case insensitive check
            if kw.lower() in source_lower:
                matched.append(kw)
            else:
                missing.append(kw)

        score = len(matched) / len(keywords)
        return score, matched, missing
