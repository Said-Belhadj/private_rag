# Private RAG

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-orange)

A **privacy-first Retrieval-Augmented Generation (RAG)** system for querying your own documents with any LLM — local or cloud — without sending your data to third parties by default.

---

## Overview

Private RAG lets you index a collection of local documents (PDFs, Word files, PowerPoint, CSV, Markdown…) into a persistent vector database and ask natural language questions about them. It uses **LlamaIndex** for the RAG pipeline, **ChromaDB** as the vector store, and **LiteLLM** as a unified LLM adapter so you can point it at Ollama (local), OpenAI, Anthropic, or any other provider by changing a single environment variable.

---

## Architecture

```
Your Documents (PDF, DOCX, PPTX, CSV, MD…)
        │
        ▼
  [core/ingestion/loader.py]  ←  loads & cleans documents
        │
        ▼
  [core/indexer.py]           ←  chunks text + embeds with HuggingFace
        │
        ▼
  ChromaDB (persistent)       ←  stores vectors on disk
        │
        ▼
  [core/engine.py]            ←  retrieves top-k chunks + calls LLM
        │
        ▼
  Answer (CLI or API)
```

---

## Features

- **Privacy-first** — works entirely offline with a local Ollama model; no data leaves your machine unless you choose a cloud LLM
- **Multi-format ingestion** — PDF, DOCX, PPTX, CSV, Markdown, HTML
- **Flexible LLM backend** — swap between Ollama, OpenAI, Anthropic, or any LiteLLM-supported provider via one env variable
- **Multi-language support** — detects the language of your query and responds in kind
- **Conversation memory** — uses `CondenseQuestionChatEngine` to maintain context across turns
- **Persistent vector store** — ChromaDB with SQLite backend; index survives restarts

---

## Tech Stack

| Component | Library | Role |
|-----------|---------|------|
| RAG framework | [LlamaIndex](https://www.llamaindex.ai/) | Orchestrates ingestion, indexing, and query |
| Vector store | [ChromaDB](https://www.trychroma.com/) | Stores and retrieves document embeddings |
| LLM adapter | [LiteLLM](https://docs.litellm.ai/) | Unified interface to all LLM providers |
| Embeddings | [HuggingFace](https://huggingface.co/) (`BAAI/bge-m3`) | Generates text embeddings locally |
| Web API | [FastAPI](https://fastapi.tiangolo.com/) | REST API layer (WIP) |
| Config | [python-dotenv](https://pypi.org/project/python-dotenv/) | Environment-based configuration |

---

## Prerequisites

- **Python 3.12+**
- **[Poetry](https://python-poetry.org/)** for dependency management
- An LLM provider of your choice:
  - **Local (default):** [Ollama](https://ollama.com/) running with your chosen model (e.g. `ollama pull qwen3:8b`)
  - **Cloud:** an OpenAI or Anthropic API key

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd private_rag

# 2. Install dependencies
poetry install

# 3. Configure your environment
cp .env.example .env
# Then edit .env to match your setup (see Configuration below)
```

---

## Configuration

All settings are controlled via the `.env` file. Copy `.env.example` to get started.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `ollama/qwen3:8b` | LiteLLM model string — e.g. `openai/gpt-4o`, `anthropic/claude-3-5-sonnet-latest` |
| `OLLAMA_API_BASE` | `http://localhost:11434` | Ollama server URL (only needed for local models) |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key (if using an OpenAI model) |
| `ANTHROPIC_API_KEY` | *(empty)* | Anthropic API key (if using a Claude model) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace embedding model — runs locally |
| `CHUNK_SIZE` | `512` | Token size of each text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap in tokens between consecutive chunks |
| `CHROMADB_PATH` | `./chroma_private_rag` | Directory where ChromaDB persists vectors |
| `CHROMADB_COLLECTION_NAME` | `private_rag_collection` | ChromaDB collection name |
| `DATA_DIR` | `./data` | Directory containing your source documents |

---

## Usage

### 1. Add your documents

Drop your files into the `data/` directory. Supported formats: `.pdf`, `.docx`, `.pptx`, `.csv`, `.md`, `.html`.

### 2. Build (or update) the vector index

```bash
poetry run python -m core.indexer
```

This loads all documents from `DATA_DIR`, embeds them, and stores the vectors in ChromaDB. Re-run whenever you add new documents.

### 3. Start the interactive chat (CLI)

```bash
poetry run python -m core.engine
```

Type your question and press Enter. The engine retrieves the most relevant chunks from your documents and generates an answer using your configured LLM. Type `quit`, `exit`, or `q` to stop.

```
You: What does the Q3 report say about revenue?
Assistant: According to the Q3 report, revenue increased by...

You: q
Goodbye!
```

### 4. Run the REST API (WIP)

```bash
poetry run uvicorn api.main:app --reload
```

> The API is a work-in-progress. Only a health-check endpoint (`GET /`) is available for now. REST endpoints for querying the RAG system are planned (see Roadmap).

---

## Project Structure

```
private_rag/
├── api/
│   ├── main.py              # FastAPI application entry point
│   └── routers.py           # API routes (WIP — empty)
├── core/
│   ├── config.py            # LLM + embedding initialisation
│   ├── engine.py            # Chat engine and interactive CLI loop
│   ├── indexer.py           # Document embedding and ChromaDB ingestion
│   └── ingestion/
│       └── loader.py        # Multi-format document loader
├── data/                    # Your documents go here (git-ignored)
├── chroma_private_rag/      # Persistent vector database (git-ignored)
├── tests/                   # Test suite (WIP)
├── .env.example             # Configuration template
└── pyproject.toml           # Project metadata and dependencies
```

---

## Roadmap

The following improvements are planned (non-exhaustive):

- [ ] **RAG & chunking improvements** — smarter chunking strategies (semantic, recursive), hybrid search (dense + sparse), re-ranking
- [ ] **REST API** — implement `api/routers.py` with query and indexing endpoints
- [ ] **Docker setup** — `docker-compose.yml` bundling the app and (optionally) an Ollama service
- [ ] **Incremental re-indexing** — detect and embed only new or changed documents, skip already-indexed ones

---

## Contributing

Contributions are welcome. Please open an issue to discuss what you'd like to change before submitting a pull request.

---

## License

MIT

Author: Saïd BELHDJ
