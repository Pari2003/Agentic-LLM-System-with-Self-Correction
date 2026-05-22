"""
PDF parsing module using pdfplumber and PyMuPDF (fitz).

Responsible for:
1. Extracting clean raw text from PDF files.
2. Detecting and extracting tables, returning them as structured text/markdown.
3. Extracting embedded images/figures, converting them to base64.
4. Maintaining page numbers and structural metadata (section titles).

Design choices:
- pdfplumber: excellent for tabular data extraction and precise text positioning.
- PyMuPDF (fitz): extremely fast text extraction and robust image extraction.
- Combined approach: PyMuPDF for quick page count/image harvesting, pdfplumber for text + tables.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Generator, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
import structlog

from src.models.schemas import ChunkType

logger = structlog.get_logger(__name__)


class PDFParser:
    """Extracts text, tables, and images from research papers/PDFs."""

    def __init__(self):
        pass

    def extract_metadata(self, pdf_path: Path) -> dict[str, Any]:
        """Extract high-level document metadata using PyMuPDF."""
        try:
            with fitz.open(pdf_path) as doc:
                metadata = doc.metadata
                return {
                    "title": metadata.get("title"),
                    "authors": metadata.get("author", "").split(";"),
                    "total_pages": len(doc),
                }
        except Exception as e:
            logger.error("pdf_metadata_extraction_failed", path=str(pdf_path), error=str(e))
            return {"title": None, "authors": [], "total_pages": 0}

    def parse(
        self,
        pdf_path: Path,
    ) -> Generator[Tuple[int, ChunkType, str, Optional[str]], None, None]:
        """Parse a PDF page-by-page, yielding raw text, tables, and images.

        Yields:
            Tuple of:
            - page_number (1-indexed int)
            - chunk_type (ChunkType.TEXT, ChunkType.TABLE, ChunkType.FIGURE_CAPTION)
            - content (str: raw text, Markdown table, or image base64)
            - section_title (Optional[str])
        """
        logger.info("pdf_parse_start", path=str(pdf_path))

        # First pass: Extract images using PyMuPDF
        images_by_page = self._extract_images(pdf_path)

        # Second pass: Extract text and tables page-by-page using pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    section_title = self._detect_section_title(page)

                    # 1. Yield any images extracted from this page
                    if page_num in images_by_page:
                        for img_base64 in images_by_page[page_num]:
                            yield page_num, ChunkType.FIGURE_CAPTION, img_base64, section_title

                    # 2. Extract tables
                    tables = page.find_tables()
                    table_bboxes = []
                    
                    for table in tables:
                        table_bboxes.append(table.bbox)
                        table_data = table.extract()
                        if table_data:
                            markdown_table = self._format_table_as_markdown(table_data)
                            yield page_num, ChunkType.TABLE, markdown_table, section_title

                    # 3. Extract text excluding table areas to avoid double-processing
                    page_text = ""
                    if table_bboxes:
                        # Extract text outside tables by filtering
                        # We sort bboxes from top to bottom
                        sorted_bboxes = sorted(table_bboxes, key=lambda b: b[1])
                        
                        last_y = 0
                        for bbox in sorted_bboxes:
                            x0, y0, x1, y1 = bbox
                            # Extract text above the table
                            crop = page.crop((0, last_y, page.width, y0))
                            text = crop.extract_text()
                            if text:
                                page_text += text + "\n"
                            last_y = y1
                        
                        # Extract remaining text below the last table
                        crop = page.crop((0, last_y, page.width, page.height))
                        text = crop.extract_text()
                        if text:
                            page_text += text
                    else:
                        page_text = page.extract_text() or ""

                    # Clean and yield page text if any exists
                    cleaned_text = self._clean_text(page_text)
                    if cleaned_text:
                        yield page_num, ChunkType.TEXT, cleaned_text, section_title

            logger.info("pdf_parse_complete", path=str(pdf_path))

        except Exception as e:
            logger.error("pdf_parse_failed", path=str(pdf_path), error=str(e))
            raise

    def _extract_images(self, pdf_path: Path) -> dict[int, list[str]]:
        """Extract all images from the PDF using PyMuPDF.

        Returns:
            Dict mapping page_number (1-indexed) to list of base64-encoded PNG strings.
        """
        images_by_page: dict[int, list[str]] = {}
        try:
            with fitz.open(pdf_path) as doc:
                for page_idx in range(len(doc)):
                    page = doc[page_idx]
                    page_num = page_idx + 1
                    image_list = page.get_images(full=True)

                    for img_idx, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Filter out very small images (icons, logos, etc.)
                        if len(image_bytes) < 5000:
                            continue

                        # Encode to base64
                        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        # Format as data URI
                        formatted_uri = f"data:image/{image_ext};base64,{img_base64}"
                        
                        if page_num not in images_by_page:
                            images_by_page[page_num] = []
                        images_by_page[page_num].append(formatted_uri)

            logger.debug("pdf_image_extraction_complete", pages_with_images=list(images_by_page.keys()))
        except Exception as e:
            logger.warning("pdf_image_extraction_error", path=str(pdf_path), error=str(e))
            
        return images_by_page

    def _clean_text(self, text: str) -> str:
        """Perform basic cleanup on extracted text.

        Removes duplicate spaces, normalizes line breaks, and discards ligatures.
        """
        if not text:
            return ""
        
        # Replace common ligatures
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
        
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            # Skip page numbers/running headers (simple heuristic)
            if line.isdigit():
                continue
            if line:
                lines.append(line)
                
        return "\n".join(lines)

    def _detect_section_title(self, page: pdfplumber.page.Page) -> Optional[str]:
        """Extract a potential section title from a page.

        Looks for bold text or uppercase lines at the top of the page.
        """
        # pdfplumber allows character-level inspection
        chars = page.chars
        if not chars:
            return None

        # Look at the first 50 characters to find the biggest font size / bold
        try:
            head_chars = chars[:100]
            # Group chars into words
            words = []
            current_word = []
            current_font = None
            current_size = 0
            
            for c in head_chars:
                if c["text"].isspace():
                    if current_word:
                        words.append(("".join(current_word), current_font, current_size))
                        current_word = []
                else:
                    current_word.append(c["text"])
                    current_font = c.get("fontname", "")
                    current_size = c.get("size", 0)
            
            if current_word:
                words.append(("".join(current_word), current_font, current_size))

            # Filter words that look like section titles (e.g. font size > 11, bold/upper)
            # Find the word sequence with the maximum size
            if words:
                max_size = max(w[2] for w in words)
                if max_size > 11.5:  # Section headings are usually larger
                    heading_words = [w[0] for w in words if abs(w[2] - max_size) < 0.5]
                    heading = " ".join(heading_words).strip()
                    # Keep it reasonably short
                    if len(heading) < 60 and not heading.startswith("http"):
                        return heading
        except Exception:
            pass

        return None

    def _format_table_as_markdown(self, table_data: list[list[Optional[str]]]) -> str:
        """Convert pdfplumber raw table extraction into clean Markdown."""
        if not table_data or not table_data[0]:
            return ""

        # Clean cell contents (replace None, remove linebreaks)
        clean_table = []
        for row in table_data:
            clean_row = []
            for cell in row:
                if cell is None:
                    clean_row.append("")
                else:
                    # Remove linebreaks and excess spaces
                    clean_row.append(str(cell).replace("\n", " ").replace("|", "\\|").strip())
            clean_table.append(clean_row)

        headers = clean_table[0]
        rows = clean_table[1:]

        # Create markdown format
        markdown = "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            # Ensure row matches header length
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))
            elif len(row) > len(headers):
                row = row[:len(headers)]
            markdown += "| " + " | ".join(row) + " |\n"

        return markdown
