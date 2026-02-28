"""Ingestion sub-package for the Private RAG system.

Exposes the public API for loading local documents from disk into
LlamaIndex :class:`~llama_index.core.Document` objects ready for
indexing.

Public API:
    load_local_documents: Load and clean documents from a local directory.
"""

from .loader import load_local_documents

__all__: list[str] = ["load_local_documents"]
