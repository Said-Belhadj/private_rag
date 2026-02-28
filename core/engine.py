"""Chat engine factory and interactive CLI loop for the Private RAG system."""

import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.vector_stores.chroma import ChromaVectorStore

from core.config import setup_settings

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

CHROMADB_PATH: str = os.getenv("CHROMADB_PATH", "./chroma_db")
CHROMADB_COLLECTION_NAME: str = os.getenv("CHROMADB_COLLECTION_NAME", "private_rag_collection")


def get_chat_engine() -> CondenseQuestionChatEngine:
    """Build and return a streaming chat engine backed by the ChromaDB index.

    Connects to the persistent ChromaDB collection, wraps it in a
    VectorStoreIndex, and wires up two custom prompts:
    - A QA prompt that enforces answering in the user's query language.
    - A condense prompt that rewrites follow-up questions as standalone
      queries while preserving the original language.

    Returns:
        A CondenseQuestionChatEngine with streaming enabled and
        similarity_top_k set to 15.

    Raises:
        Exception: If the ChromaDB collection does not exist yet
            (i.e., build_index() has not been run).
    """
    setup_settings()

    db = chromadb.PersistentClient(path=CHROMADB_PATH)
    chroma_collection = db.get_collection(name=CHROMADB_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index: VectorStoreIndex = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    qa_prompt_str: str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "CRITICAL INSTRUCTION: You must detect the language of the user's Query and write your final Answer in that EXACT SAME LANGUAGE, regardless of the language used in the context.\n"
        "If the answer is not in the context, simply state that you don't know.\n\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt: PromptTemplate = PromptTemplate(qa_prompt_str)

    condense_prompt_str: str = (
        "Given the following conversation between a Human and an AI Assistant, and a follow-up message from the Human, "
        "rewrite the follow-up message to be a standalone question that captures all relevant context from the conversation.\n"
        "IMPORTANT: The standalone question must be written in the EXACT SAME LANGUAGE as the follow-up message.\n\n"
        "<Chat History>\n"
        "{chat_history}\n\n"
        "<Follow Up Message>\n"
        "{question}\n\n"
        "<Standalone question>:"
    )
    condense_prompt: PromptTemplate = PromptTemplate(condense_prompt_str)

    query_engine = index.as_query_engine(
        similarity_top_k=15,
        text_qa_template=qa_prompt,
        streaming=True,
    )

    chat_engine: CondenseQuestionChatEngine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        condense_question_prompt=condense_prompt,
        chat_history=[],
        verbose=True,
    )

    return chat_engine


def chat_loop() -> None:
    """Run the interactive CLI chat loop against the RAG engine.

    Initialises the chat engine, then enters a read-eval-print loop that
    streams LLM responses token by token.  Type ``quit``, ``exit``, or
    ``q`` to terminate the session.
    """
    try:
        engine: CondenseQuestionChatEngine = get_chat_engine()
    except Exception as e:
        print(f"Error: ({e})")
        return

    print("\n" + "=" * 50)
    print("(Private RAG) !")
    print("Ask question to your RAG.")
    print("=" * 50)

    while True:
        user_input: str = input("\n👤 You : ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Bye !")
            break

        if not user_input.strip():
            continue

        print("AI : ", end="", flush=True)

        response_stream = engine.stream_chat(user_input)

        print("AI : ", end="", flush=True)

        for token in response_stream.response_gen:
            print(token, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    chat_loop()
