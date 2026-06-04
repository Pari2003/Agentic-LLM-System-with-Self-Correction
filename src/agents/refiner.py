"""
Refiner Agent.

Responsible for correcting and polishing LLM responses that failed the Critic's check.
Uses detailed claim-level critique reports to revise answers and remove hallucinations.
"""

from __future__ import annotations

import re

from src.agents.base import BaseAgent
from src.models.llm_client import OllamaClient
from src.models.schemas import Citation, CritiqueResult, RankedContext
from src.storage.document_store import DocumentStore


class Refiner(BaseAgent):
    """Refiner Agent that revises answers based on CritiqueResult feedback."""

    def __init__(self, llm_client: OllamaClient, doc_store: DocumentStore):
        super().__init__(llm_client)
        self.doc_store = doc_store

    async def refine(
        self,
        query: str,
        original_answer: str,
        critique_result: CritiqueResult,
        contexts: list[RankedContext],
    ) -> tuple[str, list[Citation]]:
        """Refine the response by fixing hallucinated claims using the critique report.

        Args:
            query: The user query.
            original_answer: The answer that failed critique.
            critique_result: The CritiqueResult containing the details of hallucinated claims.
            contexts: List of RankedContext parent chunks retrieved for context.

        Returns:
            A tuple of (refined_answer_text, list_of_citations).
        """
        start_time = self.start_timer()
        self.logger.info(
            "refinement_start",
            query=query,
            hallucinated_claims_count=len(critique_result.hallucinated_claims),
        )

        if not contexts:
            answer = "I could not find any relevant passages in the uploaded documents to answer your question."
            self.stop_timer_and_log("refinement", start_time, citations_count=0)
            return answer, []

        # 1. Format source contexts
        formatted_passages = []
        doc_cache = {}

        for idx, ctx in enumerate(contexts):
            source_id = idx + 1
            doc_id = ctx.parent_chunk.document_id

            if doc_id not in doc_cache:
                doc_meta = self.doc_store.get_document(doc_id)
                doc_cache[doc_id] = (
                    doc_meta.title
                    if (doc_meta and doc_meta.title)
                    else doc_meta.filename
                    if doc_meta
                    else f"Doc {doc_id[:8]}"
                )

            doc_title = doc_cache[doc_id]
            section = ctx.parent_chunk.section_title or "General"
            pages = (
                ", ".join(map(str, ctx.parent_chunk.page_numbers))
                if ctx.parent_chunk.page_numbers
                else "N/A"
            )

            formatted_passages.append(
                f"Source [{source_id}] (Document: '{doc_title}', Section: '{section}', Pages: {pages}):\n"
                f"{ctx.parent_chunk.text}"
            )

        context_str = "\n\n---\n\n".join(formatted_passages)

        # 2. Format hallucinated claims with their explanations
        hallucinated_lines = []
        for idx, v in enumerate(critique_result.hallucinated_claims):
            hallucinated_lines.append(
                f'{idx + 1}. Factual Claim: "{v.claim.text}"\n'
                f"   - Error description: {v.explanation}\n"
                f'   - Original sentence: "{v.claim.source_sentence}"'
            )
        hallucinated_claims_str = "\n".join(hallucinated_lines)

        # 3. Build refinement prompt
        system_prompt = (
            "You are an expert AI research assistant. Your task is to REFINE and CORRECT an initial answer "
            "so that all facts are strictly grounded in the provided source contexts.\n"
            "Every time you state a fact from a passage, you must immediately cite it with the bracketed "
            "source number, e.g., [1] or [2].\n"
            "Do not invent, speculate, or extrapolate. If the context does not contain enough information, "
            "remove the unverified claims entirely.\n"
            "Ensure the final response is highly objective, accurate, and completely resolved of any hallucinations."
        )

        user_prompt = (
            f"Source contexts (use ONLY these to verify facts):\n"
            f"======================\n"
            f"{context_str}\n"
            f"======================\n\n"
            f"Query: {query}\n\n"
            f"Initial Answer (contains errors):\n"
            f"----------------------\n"
            f"{original_answer}\n"
            f"----------------------\n\n"
            f"Feedback on Hallucinated Claims (MUST be corrected or deleted):\n"
            f"----------------------\n"
            f"{hallucinated_claims_str}\n"
            f"----------------------\n\n"
            f"Please generate the corrected and refined answer:"
        )

        # 4. Generate refined text (temperature: 0.1 for high grounding)
        refined_answer = await self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )

        # 5. Extract citations in the refined response
        citation_matches = sorted(list(set(re.findall(r"\[(\d+)\]", refined_answer))))
        citations = []

        for match in citation_matches:
            source_id = int(match)
            ctx_idx = source_id - 1
            if 0 <= ctx_idx < len(contexts):
                ctx = contexts[ctx_idx]
                doc_id = ctx.parent_chunk.document_id
                doc_title = doc_cache.get(doc_id, f"Doc {doc_id[:8]}")
                page_number = (
                    ctx.parent_chunk.page_numbers[0] if ctx.parent_chunk.page_numbers else None
                )

                citations.append(
                    Citation(
                        source_id=source_id,
                        chunk_id=ctx.parent_chunk.id,
                        document_id=doc_id,
                        document_title=doc_title,
                        section_title=ctx.parent_chunk.section_title,
                        page_number=page_number,
                        relevant_text=ctx.parent_chunk.text,
                    )
                )

        self.stop_timer_and_log(
            "refinement",
            start_time,
            citations_count=len(citations),
            answer_length=len(refined_answer),
        )

        return refined_answer, citations
