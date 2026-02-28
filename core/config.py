"""LLM and embedding configuration loaded from environment variables."""

import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.litellm import LiteLLM

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

LLM_MODEL: str = os.getenv("LLM_MODEL", "ollama/mistral")
OLLAMA_API_BASE: str = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))


def setup_llm() -> None:
    """Initialise Settings.llm via LiteLLM according to LLM_MODEL.

    Routes Ollama models to OLLAMA_API_BASE; all other providers rely
    on LiteLLM's default routing (API key from environment).

    Side Effects:
        Sets llama_index.core.Settings.llm globally.
    """
    if LLM_MODEL.startswith("ollama/"):
        Settings.llm = LiteLLM(
            model=LLM_MODEL,
            api_base=OLLAMA_API_BASE,
            timeout=120.0,
        )
    else:
        Settings.llm = LiteLLM(
            model=LLM_MODEL,
            timeout=120.0,
        )


def setup_settings() -> None:
    """Configure LlamaIndex global Settings for the RAG pipeline.

    Sets the node parser (SentenceSplitter), embedding model
    (HuggingFaceEmbedding), and LLM (via setup_llm).

    Side Effects:
        Modifies llama_index.core.Settings globally.
    """
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    setup_llm()
