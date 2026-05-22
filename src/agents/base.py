"""
Base Agent class.

Provides common utilities, structured logging, and latency timing helpers
for all pipeline agents (QueryAnalyzer, Retriever, Generator, Critic, Refiner).
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from src.models.llm_client import OllamaClient


class BaseAgent:
    """Foundational agent class wrapping the Ollama client and logging helpers."""

    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        # Instantiate logger specific to the inheriting subclass
        self.logger = structlog.get_logger(self.__class__.__name__)

    def start_timer(self) -> float:
        """Start a high-resolution execution timer."""
        return time.perf_counter()

    def stop_timer_and_log(self, event_name: str, start_time: float, **kwargs: Any) -> float:
        """Stop the timer, compute duration in milliseconds, and log structural attributes.

        Returns:
            Elapsed time in milliseconds.
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.logger.info(
            f"{event_name}_complete",
            elapsed_ms=round(elapsed_ms, 1),
            **kwargs,
        )
        return elapsed_ms
