"""
Async Ollama API client with retry logic, exponential backoff, and health checks.

Wraps the Ollama REST API for three capabilities:
1. Text generation (Llama 3.2 3B) — used by all agents
2. Embedding generation (nomic-embed-text) — used by ingestion + retrieval
3. Vision captioning (Moondream2) — used by image captioner during ingestion

Design decisions:
- Async (httpx) for non-blocking I/O in the FastAPI event loop
- Exponential backoff on retries (Ollama can be slow to load models)
- JSON mode for structured output (entity extraction, claim extraction)
- Lazy client initialization (no connection until first request)
- Graceful health check that reports model availability

Usage:
    from src.models.llm_client import OllamaClient

    client = OllamaClient()
    response = await client.generate("What is attention?")
    embeddings = await client.embed(["sentence 1", "sentence 2"])
    health = await client.health_check()
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import httpx
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class OllamaClient:
    """Async client for the Ollama REST API with retry logic and health checks."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        text_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        self.base_url = base_url or settings.ollama_base_url
        self.text_model = text_model or settings.text_model
        self.embed_model = embed_model or settings.embed_model
        self.vision_model = vision_model or settings.vision_model
        self.timeout = timeout or settings.llm_timeout
        self.max_retries = max_retries or settings.llm_max_retries
        self._client: Optional[httpx.AsyncClient] = None

    # ─── Connection Management ────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize the HTTP client on first use."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client connection. Call during application shutdown."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("ollama_client_closed")

    # ─── Internal Request Handler ─────────────────────────────────────────

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send HTTP request with exponential backoff retry logic.

        Retries on:
        - Connection errors (Ollama server not ready)
        - Timeout errors (model loading takes time)
        - HTTP 5xx errors (server overloaded)

        Does NOT retry on:
        - HTTP 4xx errors (bad request — our bug)
        """
        client = await self._get_client()
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx)
                if e.response.status_code < 500:
                    logger.error(
                        "ollama_client_error",
                        status_code=e.response.status_code,
                        url=url,
                        detail=e.response.text[:500],
                    )
                    raise
                last_error = e

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e

            # Exponential backoff: 2s, 4s, 8s
            if attempt < self.max_retries:
                wait = 2**attempt
                logger.warning(
                    "ollama_request_retry",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    wait_seconds=wait,
                    error=str(last_error),
                    url=url,
                )
                await asyncio.sleep(wait)

        raise ConnectionError(
            f"Ollama request to {url} failed after {self.max_retries} attempts: {last_error}"
        )

    # ─── Text Generation ──────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate text using the LLM.

        Args:
            prompt: The user prompt to send.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens: Maximum tokens to generate.
            model: Override the default text model.

        Returns:
            Generated text string.
        """
        model = model or self.text_model
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        start = time.perf_counter()
        logger.debug("ollama_generate_start", model=model, prompt_len=len(prompt))

        response = await self._request_with_retry("POST", "/api/generate", json=payload)
        result = response.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        generated_text = result.get("response", "")

        logger.info(
            "ollama_generate_complete",
            model=model,
            response_len=len(generated_text),
            eval_count=result.get("eval_count", 0),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return generated_text

    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generate structured JSON output from the LLM.

        Uses Ollama's JSON mode to constrain output to valid JSON.
        Falls back to an empty dict if parsing fails.

        Args:
            prompt: The user prompt (should instruct JSON output format).
            system_prompt: Optional system instruction.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Parsed JSON as a Python dict.
        """
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens

        payload: dict[str, Any] = {
            "model": self.text_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        start = time.perf_counter()
        logger.debug("ollama_generate_json_start", prompt_len=len(prompt))

        response = await self._request_with_retry("POST", "/api/generate", json=payload)
        result = response.json()
        raw_response = result.get("response", "")

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug("ollama_generate_json_raw", elapsed_ms=round(elapsed_ms, 1))

        try:
            parsed = json.loads(raw_response)
            logger.info(
                "ollama_generate_json_complete",
                keys=list(parsed.keys()) if isinstance(parsed, dict) else "non-dict",
                elapsed_ms=round(elapsed_ms, 1),
            )
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except json.JSONDecodeError:
            logger.error(
                "ollama_json_parse_error",
                raw_response_preview=raw_response[:300],
            )
            return {}

    # ─── Embedding Generation ─────────────────────────────────────────────

    async def embed(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Uses Ollama's batch embedding endpoint for efficiency.
        nomic-embed-text produces 768-dimensional vectors.

        Args:
            texts: List of text strings to embed.
            model: Override the default embedding model.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        model = model or self.embed_model

        payload = {
            "model": model,
            "input": texts,
        }

        start = time.perf_counter()
        logger.debug("ollama_embed_start", model=model, num_texts=len(texts))

        response = await self._request_with_retry("POST", "/api/embed", json=payload)
        result = response.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        embeddings = result.get("embeddings", [])

        logger.info(
            "ollama_embed_complete",
            model=model,
            num_embeddings=len(embeddings),
            dim=len(embeddings[0]) if embeddings else 0,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text string.

        Convenience wrapper around embed() for single-text use cases.
        """
        embeddings = await self.embed([text])
        return embeddings[0]

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> list[list[float]]:
        """Generate embeddings in batches to avoid overwhelming the server.

        For large document ingestion, sends texts in chunks of batch_size.

        Args:
            texts: All texts to embed.
            batch_size: Number of texts per batch request.

        Returns:
            All embedding vectors, preserving input order.
        """
        batch_size = batch_size or settings.embedding_batch_size
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await self.embed(batch)
            all_embeddings.extend(batch_embeddings)

            logger.debug(
                "ollama_embed_batch_progress",
                completed=min(i + batch_size, len(texts)),
                total=len(texts),
            )

        return all_embeddings

    # ─── Vision / Image Captioning ────────────────────────────────────────

    async def caption_image(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate a caption for an image using the vision model (Moondream2).

        Called during PDF ingestion to convert figures/charts into searchable text.
        The vision model is loaded on-demand by Ollama and automatically unloaded
        when other models are requested (saves RAM on 8GB systems).

        Args:
            image_base64: Base64-encoded image data.
            prompt: Custom prompt for the vision model.

        Returns:
            Generated image caption/description.
        """
        prompt = prompt or (
            "Describe this figure from a research paper in detail. "
            "Include any data, labels, axes, trends, or key findings shown. "
            "If it contains a table, transcribe the data."
        )

        payload: dict[str, Any] = {
            "model": self.vision_model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 512,
            },
        }

        start = time.perf_counter()
        logger.debug("ollama_vision_caption_start", model=self.vision_model)

        response = await self._request_with_retry("POST", "/api/generate", json=payload)
        result = response.json()

        elapsed_ms = (time.perf_counter() - start) * 1000
        caption = result.get("response", "")

        logger.info(
            "ollama_vision_caption_complete",
            model=self.vision_model,
            caption_len=len(caption),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return caption

    # ─── Health Check ─────────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        """Check Ollama server health and report available models.

        Returns a status dict indicating:
        - Whether the server is reachable
        - Which models are available
        - Whether the required models (text, embed, vision) are ready

        Used by the /health API endpoint and startup verification.
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            models_data = response.json()

            available_models = [m["name"] for m in models_data.get("models", [])]

            # Check if required models are available (partial match for version tags)
            text_ready = any(self.text_model in m for m in available_models)
            embed_ready = any(self.embed_model in m for m in available_models)
            vision_ready = any(self.vision_model in m for m in available_models)

            status = {
                "status": "healthy",
                "ollama_url": self.base_url,
                "available_models": available_models,
                "text_model": {
                    "name": self.text_model,
                    "ready": text_ready,
                },
                "embed_model": {
                    "name": self.embed_model,
                    "ready": embed_ready,
                },
                "vision_model": {
                    "name": self.vision_model,
                    "ready": vision_ready,
                },
            }

            if not text_ready:
                logger.warning("text_model_not_available", model=self.text_model)
            if not embed_ready:
                logger.warning("embed_model_not_available", model=self.embed_model)

            return status

        except Exception as e:
            logger.error("ollama_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "ollama_url": self.base_url,
                "available_models": [],
                "text_model": {"name": self.text_model, "ready": False},
                "embed_model": {"name": self.embed_model, "ready": False},
                "vision_model": {"name": self.vision_model, "ready": False},
            }
