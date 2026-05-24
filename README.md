# Agentic RAG QA System with Self-Correction

An advanced, production-grade **Agentic Retrieval-Augmented Generation (RAG)** pipeline designed to minimize LLM hallucinations and maximize response correctness. The system implements a robust **Retrieve → Generate → Critique → Refine** loop, powered by a multi-layered verification framework (Semantic Embeddings, NLI-based Entailment, and Keyword overlap).

---

## 🚀 Key Features

* **Multi-Tier Storage System**: 
  * **ChromaDB**: Partitioned, session-isolated vector database for high-performance semantic search.
  * **Neo4j Graph Database**: Structure and track semantic entity-relationship maps extracted from documents.
  * **SQLite Database**: Relational engine for structured metadata, session state, and parent chunks, with Full-Text Search (FTS5) enabled.
* **Three-Stage Hybrid Retrieval**:
  * **Stage 1 (Hybrid Search)**: Combines ChromaDB vector search and SQLite BM25 text search via Reciprocal Rank Fusion (RRF) with graph-based entity boosts.
  * **Stage 2 (Parent Chunk Expansion)**: Maps retrieved child chunks (~256 tokens) to parent chunks (~1024 tokens) to supply rich contextual boundaries to the LLM.
  * **Stage 3 (LLM Reranking)**: Uses a reranker model to select the final high-priority parent contexts.
* **Orchestrated Agentic Self-Correction**:
  * **Query Analyzer**: Evaluates query complexity and reformulates the prompt for semantic matching.
  * **Critic Agent**: Validates generated claims using embedding similarities, entailment judges, and keyword metrics.
  * **Refiner Agent**: Consumes critique feedback to selectively rewrite factual errors without degrading correct text.
* **Complete Session Management**: Automatic sliding-window TTLs, background purges, and SQLite-cascaded deletions.

---

## 🛠️ Technology Stack

* **Language**: Python 3.11+
* **Vector Store**: ChromaDB (v0.5.0+)
* **Graph Database**: Neo4j (v5 Community)
* **Metadata/Search**: SQLite (FTS5)
* **LLM Engine**: Ollama (for offline execution) or any compatible OpenAI/HuggingFace client
* **API Framework**: FastAPI + Pydantic v2 + Uvicorn
* **Orchestration**: Custom async state pipeline

---

## 🗺️ System Architecture

For a detailed visual guide and breakdown of the pipeline, see [docs/architecture.md](file:///c:/Users/maitr/OneDrive/Desktop/projects/Agentic-LLM-System-with-Self-Correction/docs/architecture.md).

---

## ⚙️ Configuration & Environment

Configuration is managed via Pydantic Settings and loaded from environment variables or a `.env` file. A comprehensive template is provided in [.env.example](file:///c:/Users/maitr/OneDrive/Desktop/projects/Agentic-LLM-System-with-Self-Correction/.env.example).

### Key Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint for the Ollama LLM Server |
| `TEXT_MODEL` | `llama3.2` | Primary LLM model for generation & agents |
| `EMBED_MODEL` | `nomic-embed-text` | Core embedding model |
| `NEO4J_URI` | `bolt://localhost:7687` | Connection string for Neo4j database |
| `NEO4J_USER` | `neo4j` | Username for Neo4j |
| `NEO4J_PASSWORD` | `password123` | Password for Neo4j |
| `CHROMADB_PATH` | `./data/chromadb` | Directory where ChromaDB is persisted |
| `SQLITE_PATH` | `./data/metadata.db` | Directory where SQLite metadata is saved |

---

## ⚡ Quick Start (Docker Compose)

The easiest way to spin up the entire stack (FastAPI Backend, Neo4j, and Ollama) is using Docker Compose:

```bash
# Build and run the services in background
docker-compose up -d --build
```

### Initializing Ollama Models
Once the containers are running, you need to pull the required models inside the Ollama container:

```bash
# Pull text and embedding models
docker exec -it ollama_rag ollama pull llama3.2
docker exec -it ollama_rag ollama pull nomic-embed-text
```

The FastAPI application will be accessible at: `http://localhost:8000`. You can view the interactive Swagger API documentation at `http://localhost:8000/docs`.

---

## 💻 Local Development Setup

If you prefer to run the components locally outside of Docker:

### 1. Prerequisites
* **Python**: Install Python 3.11 or higher.
* **Ollama**: Download and install [Ollama](https://ollama.com). Pull models:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```
* **Neo4j**: Run Neo4j locally or in a standalone container:
  ```bash
  docker run -d --name neo4j_local -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:5-community
  ```

### 2. Set Up Virtual Environment
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Backend Server
```bash
# Start API using the entrypoint script
python run.py
```

---

## 📝 API Endpoints

### 1. Health & Readiness
* `GET /api/v1/health/live` — Returns liveness status.
* `GET /api/v1/health/ready` — Verifies downstream database connections (SQLite, Vector, Neo4j).

### 2. Session Management
* `POST /api/v1/sessions` — Creates a new sliding-window session.
* `GET /api/v1/sessions/{session_id}` — Retrieves session metadata.
* `DELETE /api/v1/sessions/{session_id}` — Deletes session and triggers cascading purges across all storage layers.

### 3. Ingestion
* `POST /api/v1/sessions/{session_id}/ingest` — Uploads and ingests a PDF research paper into the session space (performs semantic chunking and graph building).

### 4. Query
* `POST /api/v1/sessions/{session_id}/query` — Executes the full three-stage hybrid retrieval search and agentic self-correction loop on a question.

---

## 🧪 Verification & Demo

To help you quickly load sample papers and verify system integration:

1. **Download Classic AI/ML Papers**:
   Use the utility script to download papers (like *Attention Is All You Need* or the original *RAG* paper) into `data/papers/`:
   ```bash
   python -m scripts.download_sample_papers
   ```

2. **Run Integration Tests**:
   Ensure your Python path includes the source directory and execute the verification test suites:
   ```bash
   # Set PYTHONPATH and run storage integration test
   $env:PYTHONPATH="."
   python -m tests.test_storage
   
   # Run retrieval integration test
   python -m tests.test_retrieval
   ```
