FROM python:3.11-slim

WORKDIR /app

# System dependencies for PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/papers data/chromadb data/eval

# Expose API port
EXPOSE 8000

# Environment variables (overridable at runtime)
ENV PYTHONPATH=/app
ENV OLLAMA_BASE_URL=http://ollama:11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/health/ready'); r.raise_for_status()" || exit 1

# Start the application
# CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["tail", "-f", "/dev/null"]
