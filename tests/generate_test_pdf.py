import fitz  # PyMuPDF
from pathlib import Path

def create_mock_pdf(output_path: Path):
    """Creates a basic PDF with text, a mock table, and a small image for RAG testing."""
    # 1. Create a new PDF document
    doc = fitz.open()
    
    # Page 1: Title and Abstract
    page1 = doc.new_page()
    rect1 = fitz.Rect(50, 50, 550, 80)
    page1.insert_textbox(rect1, "Agentic LLM Systems: A New Paradigm", fontsize=18, fontname="hebo")
    
    rect2 = fitz.Rect(50, 100, 550, 200)
    abstract_text = (
        "Abstract—Agentic retrieval-augmented generation (RAG) represents a significant shift from "
        "passive context injection to active context selection. In this work, we propose a multi-stage "
        "retrieval pipeline coupled with self-correction feedback loops. Our evaluation shows a 25% "
        "improvement in reliability compared to baseline architectures. We utilize semantic chunking "
        "and light-weight GraphRAG to capture structured entity relationships."
    )
    page1.insert_textbox(rect2, abstract_text, fontsize=11, fontname="helv")
    
    # Page 2: Methodology and Table
    page2 = doc.new_page()
    rect3 = fitz.Rect(50, 50, 550, 80)
    page2.insert_textbox(rect3, "Methodology and Architecture", fontsize=14, fontname="hebo")
    
    # Text
    rect4 = fitz.Rect(50, 100, 550, 200)
    method_text = (
        "The architecture consists of three components. First, the ingestion pipeline splits text into "
        "semantic units using sentence similarity. Second, the hybrid retrieval module combines dense vectors "
        "with sparse BM25 scores. Third, a self-correction loop verifies all generated claims. "
        "Table 1 outlines performance benchmarks of various models on the QA task."
    )
    page2.insert_textbox(rect4, method_text, fontsize=11, fontname="helv")
    
    # Draw a table visually (lines and text)
    # Header
    page2.insert_text(fitz.Point(70, 230), "Model", fontname="hebo", fontsize=10)
    page2.insert_text(fitz.Point(170, 230), "Accuracy (%)", fontname="hebo", fontsize=10)
    page2.insert_text(fitz.Point(270, 230), "Latency (ms)", fontname="hebo", fontsize=10)
    
    # Row 1
    page2.insert_text(fitz.Point(70, 250), "Llama 3 8B", fontname="helv", fontsize=10)
    page2.insert_text(fitz.Point(170, 250), "78.4", fontname="helv", fontsize=10)
    page2.insert_text(fitz.Point(270, 250), "450", fontname="helv", fontsize=10)
    
    # Row 2
    page2.insert_text(fitz.Point(70, 270), "GPT-4o", fontname="helv", fontsize=10)
    page2.insert_text(fitz.Point(170, 270), "92.1", fontname="helv", fontsize=10)
    page2.insert_text(fitz.Point(270, 270), "850", fontname="helv", fontsize=10)
    
    # Table grid lines
    shape = page2.new_shape()
    shape.draw_line(fitz.Point(50, 220), fitz.Point(400, 220)) # Top border
    shape.draw_line(fitz.Point(50, 238), fitz.Point(400, 238)) # Header separator
    shape.draw_line(fitz.Point(50, 280), fitz.Point(400, 280)) # Bottom border
    shape.commit()
    
    # Save the document
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()
    print(f"Mock PDF successfully created at {output_path}")

if __name__ == "__main__":
    create_mock_pdf(Path("data/papers/sample_paper.pdf"))
