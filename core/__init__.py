"""Core package for the Private RAG system.

This package provides the main building blocks of the RAG pipeline:

- :mod:`core.config`     — LLM and embedding configuration via environment variables
- :mod:`core.indexer`    — Document indexing and ChromaDB persistence
- :mod:`core.engine`     — Chat engine factory and interactive CLI loop
- :mod:`core.ingestion`  — Multi-format document loading and cleaning
"""
