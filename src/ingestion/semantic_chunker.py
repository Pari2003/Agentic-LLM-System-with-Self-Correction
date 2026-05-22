"""
Semantic chunking using sentence embeddings similarity.

Algorithm:
1. Split input text into sentences using regular expressions.
2. Group very short sentences (e.g. < 15 chars) with their neighbors.
3. Compute embeddings for all sentence groups.
4. Calculate the cosine similarity between each consecutive pair of sentence group embeddings.
5. Identify semantic boundaries where the similarity drops below a percentile threshold (default: 25%).
6. Form child chunks at these boundaries, checking token length limits (tiktoken).
7. If a child chunk is too small, merge it. If it is too large, split it.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import structlog
import tiktoken

from src.config import settings

logger = structlog.get_logger(__name__)


class SemanticChunker:
    """Chunks text based on embedding similarity drops between consecutive sentences."""

    def __init__(
        self,
        min_tokens: int = settings.child_chunk_min_tokens,
        max_tokens: int = settings.child_chunk_max_tokens,
        split_percentile: int = settings.semantic_split_percentile,
        encoding_name: str = "cl100k_base",
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.split_percentile = split_percentile
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string."""
        return len(self.tokenizer.encode(text))

    def split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using simple regex heuristic."""
        # Split on period/question/exclamation followed by space and uppercase letter,
        # while avoiding common abbreviations (e.g., e.g., i.e., Al., Fig., Vol.)
        sentence_end = re.compile(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s+(?=[A-Z0-9])'
        )
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _group_short_sentences(self, sentences: list[str]) -> list[str]:
        """Merge short sentences with adjacent ones to make embeddings reliable."""
        grouped: list[str] = []
        temp_group: list[str] = []
        
        for s in sentences:
            temp_group.append(s)
            combined = " ".join(temp_group)
            # If combined is at least 15 words or 80 chars, save it
            if len(combined.split()) >= 15 or len(combined) >= 80:
                grouped.append(combined)
                temp_group = []
                
        if temp_group:
            if grouped:
                grouped[-1] += " " + " ".join(temp_group)
            else:
                grouped.append(" ".join(temp_group))
                
        return grouped

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        arr1, arr2 = np.array(v1), np.array(v2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(arr1, arr2) / (norm1 * norm2))

    def chunk_text(
        self,
        text: str,
        embeddings: list[list[float]],  # Precomputed sentence group embeddings
        sentence_groups: list[str],     # The sentence groups matching the embeddings
    ) -> list[str]:
        """Chunk text using precomputed embeddings for the sentence groups.

        Args:
            text: The full text (should correspond to the sentence_groups).
            embeddings: Embeddings for each item in sentence_groups.
            sentence_groups: Grouped sentences corresponding to the embeddings.

        Returns:
            List of chunked text strings.
        """
        if not sentence_groups:
            return []
        if len(sentence_groups) == 1:
            return sentence_groups

        assert len(sentence_groups) == len(embeddings), "Sentence groups and embeddings count must match."

        # 1. Compute similarities between adjacent sentence group embeddings
        similarities: list[float] = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # 2. Find threshold representing bottom N percentile of similarity
        if similarities:
            threshold = float(np.percentile(similarities, self.split_percentile))
        else:
            threshold = 0.5

        logger.debug(
            "semantic_chunking_thresholds",
            percentile=self.split_percentile,
            threshold=threshold,
            min_sim=min(similarities) if similarities else 0,
            max_sim=max(similarities) if similarities else 0,
        )

        # 3. Create splits where similarity is below threshold
        chunks: list[str] = []
        current_chunk_sentences: list[str] = [sentence_groups[0]]

        for i in range(len(similarities)):
            sim = similarities[i]
            next_sentence = sentence_groups[i + 1]

            # If similarity is low, trigger a split boundary
            if sim < threshold:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [next_sentence]
            else:
                current_chunk_sentences.append(next_sentence)

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        # 4. Enforce Token Constraints: Merge small, split large
        final_chunks = self._enforce_token_constraints(chunks)
        return final_chunks

    def _enforce_token_constraints(self, raw_chunks: list[str]) -> list[str]:
        """Iterate through chunks to ensure they fit min/max token sizes."""
        constrained: list[str] = []
        current_buffer: list[str] = []
        
        for chunk in raw_chunks:
            chunk_tokens = self.count_tokens(chunk)
            
            # If a single chunk is larger than max_tokens, split it naively/sentence-wise
            if chunk_tokens > self.max_tokens:
                # Flush the buffer first
                if current_buffer:
                    constrained.append(" ".join(current_buffer))
                    current_buffer = []
                
                # Split large chunk into sub-chunks
                sub_sentences = self.split_into_sentences(chunk)
                sub_buffer: list[str] = []
                for s in sub_sentences:
                    sub_buffer.append(s)
                    if self.count_tokens(" ".join(sub_buffer)) >= self.max_tokens:
                        constrained.append(" ".join(sub_buffer))
                        sub_buffer = []
                if sub_buffer:
                    constrained.append(" ".join(sub_buffer))
                continue

            # Check if adding to the current buffer exceeds max_tokens
            combined_buffer_test = " ".join(current_buffer + [chunk])
            test_tokens = self.count_tokens(combined_buffer_test)

            if test_tokens <= self.max_tokens:
                current_buffer.append(chunk)
            else:
                # Flush buffer
                if current_buffer:
                    constrained.append(" ".join(current_buffer))
                current_buffer = [chunk]

        # Flush any remaining buffer
        if current_buffer:
            constrained.append(" ".join(current_buffer))

        # Perform a clean up pass: merge any tiny chunks that are below min_tokens
        merged_constrained: list[str] = []
        temp_merge: list[str] = []
        
        for chunk in constrained:
            temp_merge.append(chunk)
            if self.count_tokens(" ".join(temp_merge)) >= self.min_tokens:
                merged_constrained.append(" ".join(temp_merge))
                temp_merge = []

        if temp_merge:
            if merged_constrained:
                merged_constrained[-1] += " " + " ".join(temp_merge)
            else:
                merged_constrained.append(" ".join(temp_merge))

        return merged_constrained

    def prepare_sentence_groups(self, text: str) -> list[str]:
        """Convenience method to split text into preprocessed sentence groups."""
        sentences = self.split_into_sentences(text)
        return self._group_short_sentences(sentences)
