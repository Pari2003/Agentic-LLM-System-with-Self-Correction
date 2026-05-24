"""
Application entry point.

Run with: python run.py
Or:       uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn

from src.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
