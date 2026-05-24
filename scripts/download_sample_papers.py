"""
Download sample ArXiv papers for demo and evaluation.

Downloads 3-5 classic AI/ML papers in PDF format to data/papers/.

Usage:
    python -m scripts.download_sample_papers
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PAPERS = [
    {
        "name": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "filename": "attention_is_all_you_need.pdf",
    },
    {
        "name": "BERT: Pre-training of Deep Bidirectional Transformers",
        "arxiv_id": "1810.04805",
        "filename": "bert.pdf",
    },
    {
        "name": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
        "arxiv_id": "2005.11401",
        "filename": "rag_original.pdf",
    },
    {
        "name": "Chain-of-Thought Prompting Elicits Reasoning",
        "arxiv_id": "2201.11903",
        "filename": "chain_of_thought.pdf",
    },
    {
        "name": "Self-RAG: Learning to Retrieve, Generate and Critique",
        "arxiv_id": "2310.11511",
        "filename": "self_rag.pdf",
    },
]


def main():
    try:
        import httpx
    except ImportError:
        print("Error: httpx is required. Install with: pip install httpx")
        sys.exit(1)

    papers_dir = Path("data/papers")
    papers_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(PAPERS)} sample papers to {papers_dir}/\n")

    for paper in PAPERS:
        output_path = papers_dir / paper["filename"]

        if output_path.exists():
            print(f"  ✓ {paper['name']} — already exists, skipping")
            continue

        url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
        print(f"  ↓ {paper['name']} ({paper['arxiv_id']})...", end=" ", flush=True)

        try:
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                response = client.get(url)
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    f.write(response.content)

                size_mb = len(response.content) / (1024 * 1024)
                print(f"✓ ({size_mb:.1f} MB)")

        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\nDone! Papers saved to {papers_dir}/")
    print("You can now ingest these papers via the API:")
    print("  POST /api/v1/sessions/{session_id}/ingest")


if __name__ == "__main__":
    main()
