import os
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore

from core.config import setup_settings
from core.ingestion import load_local_documents

load_dotenv()

CHROMADB_PATH = os.getenv("CHROMADB_PATH", "")
CHROMADB_COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION_NAME", "")

DATA_DIR = os.getenv("DATA_DIR", "../data")

def setup_db(documents: list[Document]):
    db = chromadb.PersistentClient(path=CHROMADB_PATH)
    chroma_collection = db.get_or_create_collection(name=CHROMADB_COLLECTION_NAME)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    return index


def build_index():
    setup_settings()
    documents = load_local_documents(DATA_DIR)
    for doc in documents:
        print(doc)
        print("\n")
    setup_db(documents)

if __name__ == "__main__":
    build_index()