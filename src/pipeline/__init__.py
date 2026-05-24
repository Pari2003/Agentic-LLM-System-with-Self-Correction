"""
Pipeline Package.

Exposes the Orchestrator which coordinates retrieval, generation, critique, and self-correction.
"""

from src.pipeline.orchestrator import Orchestrator

__all__ = ["Orchestrator"]
