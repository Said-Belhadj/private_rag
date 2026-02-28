import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "ollama/mistral")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def setup_llm():
    if LLM_MODEL.startswith("ollama/"):
        Settings.llm = LiteLLM(
            model=LLM_MODEL,
            api_base=OLLAMA_API_BASE,
            timeout=120.0
        )
    else:
        Settings.llm = LiteLLM(
            model=LLM_MODEL,
            timeout=120.0
        )

def setup_settings():
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    setup_llm()
    