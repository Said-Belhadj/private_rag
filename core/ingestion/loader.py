"""Multi-format document loader for the Private RAG ingestion pipeline."""

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.file import (
    DocxReader,
    MarkdownReader,
    PandasCSVReader,
    PptxReader,
    UnstructuredReader,
)


def load_local_documents(input_dir: str) -> list[Document]:
    """Load documents from a local directory using format-specific readers.

    Supports .pdf, .docx, .pptx, .csv, .md, and .html files.
    After loading, metadata is sanitised via clean_data().

    Args:
        input_dir: Path to the directory containing source documents.

    Returns:
        A list of LlamaIndex Document objects ready for indexing.
    """
    pdf_html_reader: UnstructuredReader = UnstructuredReader()
    docx_reader: DocxReader = DocxReader()
    pptx_reader: PptxReader = PptxReader()
    csv_reader: PandasCSVReader = PandasCSVReader()
    md_reader: MarkdownReader = MarkdownReader()

    file_extractor: dict[str, object] = {
        ".pdf": pdf_html_reader,
        ".docx": docx_reader,
        ".pptx": pptx_reader,
        ".csv": csv_reader,
        ".md": md_reader,
        ".html": pdf_html_reader,
    }

    reader = SimpleDirectoryReader(
        input_dir="data/",
        recursive=True,
        required_exts=[".pdf", ".docx", ".pptx", ".csv", ".md", ".html"],
        file_extractor=file_extractor,
    )

    documents: list[Document] = reader.load_data()
    documents = clean_data(documents)

    return documents


def clean_data(documents: list[Document]) -> list[Document]:
    """Sanitise document metadata so all values are ChromaDB-serialisable.

    ChromaDB only accepts str, int, float, or None as metadata values.
    Any other type is converted to its string representation.

    Args:
        documents: List of raw Document objects returned by a reader.

    Returns:
        The same list with metadata dicts containing only primitive values.
    """
    allowed_types: tuple[type, ...] = (str, int, float, type(None))
    for doc in documents:
        doc.metadata = {
            k: (v if isinstance(v, allowed_types) else str(v))
            for k, v in doc.metadata.items()
        }
    return documents
