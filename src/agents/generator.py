"""
Generator Agent.

Responsible for drafting citation-backed answers based on retrieved parent contexts.
Each citation in the text (e.g. [1]) is parsed and returned as a structured Citation model.
"""

from __future__ import annotations

import re

from src.agents.base import BaseAgent
from src.models.llm_client import OllamaClient
from src.models.schemas import Citation, RankedContext
from src.storage.document_store import DocumentStore


class Generator(BaseAgent):
    """Agent responsible for drafting citation-backed answers based on retrieved parent contexts."""

    def __init__(self, llm_client: OllamaClient, doc_store: DocumentStore):
        super().__init__(llm_client)
        self.doc_store = doc_store

    async def generate(
        self,
        query: str,
        contexts: list[RankedContext],
        session_id: str,
    ) -> tuple[str, list[Citation]]:
        """Draft a citation-backed response based on retrieved contexts.

        Args:
            query: The user's search query.
            contexts: List of RankedContext elements (top contexts from retrieval).
            session_id: The active session ID.

        Returns:
            A tuple of (generated_answer_text, list_of_citations).
        """
        start_time = self.start_timer()
        self.logger.info("generation_start", query=query, contexts_count=len(contexts))

        if not contexts:
            answer = "I could not find any relevant passages in the uploaded documents to answer your question."
            self.stop_timer_and_log("generation", start_time, citations_count=0)
            return answer, []

        # 1. Format the context passages for the LLM
        formatted_passages = []
        doc_cache = {}  # Cache document lookups (to avoid querying SQLite multiple times)

        for idx, ctx in enumerate(contexts):
            source_id = idx + 1
            doc_id = ctx.parent_chunk.document_id
            
            # Fetch document title from cache or DB
            if doc_id not in doc_cache:
                doc_meta = self.doc_store.get_document(doc_id)
                doc_cache[doc_id] = doc_meta.title if (doc_meta and doc_meta.title) else doc_meta.filename if doc_meta else f"Doc {doc_id[:8]}"
            
            doc_title = doc_cache[doc_id]
            section = ctx.parent_chunk.section_title or "General"
            pages = ", ".join(map(str, ctx.parent_chunk.page_numbers)) if ctx.parent_chunk.page_numbers else "N/A"
            
            formatted_passages.append(
                f"Source [{source_id}] (Document: '{doc_title}', Section: '{section}', Pages: {pages}):\n"
                f"{ctx.parent_chunk.text}"
            )

        context_str = "\n\n---\n\n".join(formatted_passages)

        # 2. Build system and user prompts
        system_prompt = (
            "You are an expert AI research assistant. Your task is to answer the user query based ONLY on the provided context passages.\n"
            "Every time you state a fact from a passage, you must immediately cite it with the bracketed source number, e.g., [1] or [2].\n"
            "Do not invent, speculate, or extrapolate. If the context does not contain enough information to answer, state that you do not have enough information, but still cite whatever partial facts you retrieve.\n"
            "Keep the answer factual, objective, and fully grounded. Do not mention sources that were not provided."
        )

        user_prompt = (
            f"Context passages:\n"
            f"======================\n"
            f"{context_str}\n"
            f"======================\n\n"
            f"Query: {query}\n\n"
            f"Answer:"
        )

        # 3. Call Ollama model (temperature: 0.1 for high grounding)
        generated_answer = await self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
        )

        # 4. Extract citations and link to documents
        # Match pattern [1], [2], etc.
        citation_matches = sorted(list(set(re.findall(r"\[(\d+)\]", generated_answer))))
        citations = []

        for match in citation_matches:
            source_id = int(match)
            ctx_idx = source_id - 1
            if 0 <= ctx_idx < len(contexts):
                ctx = contexts[ctx_idx]
                doc_id = ctx.parent_chunk.document_id
                doc_title = doc_cache.get(doc_id, f"Doc {doc_id[:8]}")
                page_number = ctx.parent_chunk.page_numbers[0] if ctx.parent_chunk.page_numbers else None
                
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
            "generation",
            start_time,
            citations_count=len(citations),
            answer_length=len(generated_answer),
        )

        return generated_answer, citations
