"""
Image Captioner Module.

Uses Ollama's vision model (Moondream2) to caption images/charts/figures found in PDF files.
If the vision model is not pulled or ready, it degrades gracefully with a generic description
to ensure the ingestion pipeline never crashes on 8GB RAM local machines.
"""

from __future__ import annotations

import structlog

from src.models.llm_client import OllamaClient

logger = structlog.get_logger(__name__)


class ImageCaptioner:
    """Generates captions/descriptions for visual content extracted from PDFs."""

    def __init__(self, llm_client: OllamaClient):
        self.client = llm_client

    async def caption(
        self,
        image_base64: str,
        page_number: int,
    ) -> str:
        """Caption a base64-encoded image using Moondream2.

        Args:
            image_base64: Base64-encoded image content (data URI format).
            page_number: Page number where image was found.

        Returns:
            A textual caption describing the contents of the image.
        """
        logger.info("image_caption_start", page=page_number)

        # 1. Clean data URI prefix if present
        clean_base64 = image_base64
        if "," in image_base64:
            clean_base64 = image_base64.split(",")[1]

        # 2. Check health of Ollama vision model first
        health = await self.client.health_check()
        vision_ready = health.get("vision_model", {}).get("ready", False)

        if not vision_ready:
            logger.warning(
                "vision_model_not_ready_degrading_gracefully",
                model=self.client.vision_model,
                page=page_number,
            )
            return (
                f"[Image/Figure extracted from page {page_number}. "
                f"Vision model '{self.client.vision_model}' is not installed or ready on local host. "
                f"Please run `ollama pull {self.client.vision_model}` to enable visual ingestion.]"
            )

        try:
            prompt = (
                "Describe this figure from a research paper in detail. "
                "Transcribe any data tables, explain charts, axes, legends, or diagrams "
                "so that it can be searched and referenced in a text-based RAG pipeline."
            )
            caption_text = await self.client.caption_image(clean_base64, prompt=prompt)
            
            logger.info("image_caption_success", page=page_number, caption_len=len(caption_text))
            return f"[Figure on Page {page_number} Description: {caption_text}]"

        except Exception as e:
            logger.error("image_caption_failed", page=page_number, error=str(e))
            # Graceful degradation fallback
            return f"[Image/Figure on page {page_number} failed to parse due to vision model timeout or error.]"
