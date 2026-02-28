from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file import UnstructuredReader, DocxReader, PptxReader, PandasCSVReader, MarkdownReader



def load_local_documents(input_dir: str) -> list[Document]:
    pdf_html_reader = UnstructuredReader()
    docx_reader = DocxReader()
    pptx_reader = PptxReader()
    csv_reader = PandasCSVReader()
    md_reader = MarkdownReader()

    file_extractor = {
        '.pdf': pdf_html_reader,
        '.docx': docx_reader,
        '.pptx': pptx_reader,
        '.csv': csv_reader,
        '.md': md_reader,
        ".html": pdf_html_reader,
    }

    reader = SimpleDirectoryReader(
        input_dir="data/", 
        recursive=True, 
        required_exts=[".pdf", ".docx", ".pptx", ".csv", ".md", ".html"], 
        file_extractor=file_extractor
    )

    documents = reader.load_data()
    documents = clean_data(documents)

    return documents

def clean_data(documents: list[Document]) -> list[Document]:
    allowed_types = (str, int, float, type(None))
    for doc in documents:
        doc.metadata = {
            k: (v if isinstance(v, allowed_types) else str(v)) 
            for k, v in doc.metadata.items()
        }
    return documents 

