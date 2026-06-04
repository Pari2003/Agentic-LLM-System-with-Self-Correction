import asyncio
from pathlib import Path

from src.ingestion.embedder import Embedder
from src.ingestion.parent_child_linker import ParentChildLinker
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.semantic_chunker import SemanticChunker
from src.models.llm_client import OllamaClient
from src.models.schemas import ChunkType


async def main():
    print("=== STARTING PHASE 2 INGESTION TEST ===")

    # 1. Initialize all components
    print("\n[1] Initializing components...")
    client = OllamaClient()
    parser = PDFParser()
    chunker = SemanticChunker(min_tokens=20, max_tokens=150)  # small limits for test paper
    linker = ParentChildLinker(parent_max_tokens=300)  # small limits for test paper
    embedder = Embedder(client)

    pdf_path = Path("data/papers/sample_paper.pdf")
    doc_id = "test_doc_123"
    session_id = "test_session_abc"

    # 2. Extract metadata
    print("\n[2] Extracting PDF metadata...")
    meta = parser.extract_metadata(pdf_path)
    print(f"Metadata extracted: {meta}")

    # 3. Parse text and tables
    print("\n[3] Parsing pages...")
    parsed_items = list(parser.parse(pdf_path))
    print(f"Parsed {len(parsed_items)} items from PDF:")
    for page_num, c_type, content, section in parsed_items:
        print(
            f"  Page {page_num} | Type: {c_type.value} | Length: {len(content)} characters | Section: {section}"
        )
        if c_type == ChunkType.TABLE:
            print("  --- Extracted Table ---")
            print(content)
            print("  -----------------------")

    # 4. Prep sentence groups for semantic chunking
    print("\n[4] Preparing text for semantic chunking...")
    # Gather all text content to chunk
    text_content = ""
    for page_num, c_type, content, section in parsed_items:
        if c_type == ChunkType.TEXT:
            text_content += content + "\n"

    sentence_groups = chunker.prepare_sentence_groups(text_content)
    print(f"Split text into {len(sentence_groups)} sentence groups:")
    for idx, group in enumerate(sentence_groups):
        print(f"  Group {idx + 1}: '{group[:70]}...' ({chunker.count_tokens(group)} tokens)")

    # 5. Generate embeddings for sentence groups to find semantic splits
    print("\n[5] Embedding sentence groups for similarity analysis...")
    group_embeddings = await client.embed(sentence_groups)
    print(
        f"Generated {len(group_embeddings)} embeddings of dimension {len(group_embeddings[0]) if group_embeddings else 0}"
    )

    # 6. Perform semantic chunking
    print("\n[6] Performing semantic chunking...")
    child_texts = chunker.chunk_text(text_content, group_embeddings, sentence_groups)
    print(f"Formed {len(child_texts)} semantic child chunks:")
    for idx, chunk in enumerate(child_texts):
        print(f"  Child Chunk {idx + 1}: '{chunk[:80]}...' ({chunker.count_tokens(chunk)} tokens)")

    # 7. Convert to Parent-Child structure
    print("\n[7] Linking child chunks into parent structures...")
    # Format for linker
    child_items_to_link = []
    # For simplicity in this test, assign them to page 1 and section "Introduction"
    for text in child_texts:
        child_items_to_link.append(
            {
                "text": text,
                "page_number": 1,
                "section_title": "Abstract/Intro",
                "chunk_type": ChunkType.TEXT,
            }
        )

    # Add the table as a separate child chunk to make sure tables are indexed
    for page_num, c_type, content, section in parsed_items:
        if c_type == ChunkType.TABLE:
            child_items_to_link.append(
                {
                    "text": content,
                    "page_number": page_num,
                    "section_title": section or "Methodology",
                    "chunk_type": ChunkType.TABLE,
                }
            )

    child_chunks, parent_chunks = linker.link(doc_id, session_id, child_items_to_link)

    print(f"\nCreated {len(parent_chunks)} Parent Chunks and {len(child_chunks)} Child Chunks.")
    for idx, parent in enumerate(parent_chunks):
        print(
            f"  Parent {idx + 1} ({parent.id}): '{parent.text[:80]}...' contains children: {parent.child_ids}"
        )

    # 8. Generate embeddings for final child chunks (stage 1 search vectors)
    print("\n[8] Embedding child chunks for vector search index...")
    embedded_children = await embedder.embed_chunks(child_chunks)
    print(f"Generated embeddings for {len(embedded_children)} child chunks.")
    print(
        f"First child chunk vector sample (first 5 elements): {embedded_children[0].embedding[:5]}"
    )

    print("\n=== PHASE 2 INGESTION PIPELINE FULLY VERIFIED! ===")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
