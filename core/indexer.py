"""Document indexing pipeline: embeds documents and persists them in ChromaDB."""

import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from core.config import setup_settings
from core.ingestion import load_local_documents

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

CHROMADB_PATH: str = os.getenv("CHROMADB_PATH", "")
CHROMADB_COLLECTION_NAME: str = os.getenv("CHROMADB_COLLECTION_NAME", "")
DATA_DIR: str = os.getenv("DATA_DIR", "../data")


def setup_db(documents: list[Document]) -> VectorStoreIndex:
    """Embed documents and persist the vector index to ChromaDB.

    Creates (or reuses) a ChromaDB collection, wraps it in a
    ChromaVectorStore, and builds a VectorStoreIndex from the provided
    documents with a progress indicator.

    Args:
        documents: List of LlamaIndex Document objects to embed and store.

    Returns:
        A VectorStoreIndex backed by the ChromaDB persistent collection.
    """
    db = chromadb.PersistentClient(path=CHROMADB_PATH)
    chroma_collection = db.get_or_create_collection(name=CHROMADB_COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    return index


def build_index() -> None:
    """Orchestrate the full indexing pipeline: load → embed → persist.

    1. Initialises LlamaIndex settings (LLM, embeddings, chunking).
    2. Loads documents from DATA_DIR using load_local_documents().
    3. Embeds and persists them to ChromaDB via setup_db().
    """
    setup_settings()
    documents: list[Document] = load_local_documents(DATA_DIR)
    for doc in documents:
        print(doc)
        print("\n")
    setup_db(documents)


if __name__ == "__main__":
    build_index()
