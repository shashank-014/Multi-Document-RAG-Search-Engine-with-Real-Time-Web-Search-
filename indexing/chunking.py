from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_OVERLAP, CHUNK_SIZE
from ingestion.schema import IngestedDocument



def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )



def chunk_documents(records: list[IngestedDocument]) -> list[Document]:
    splitter = build_splitter()
    chunks: list[Document] = []

    for record in records:
        text_chunks = splitter.split_text(record.content)
        for chunk_index, text in enumerate(text_chunks):
            metadata = {
                **record.metadata,
                "source_id": record.source_id,
                "source_type": record.source_type,
                "title": record.title,
                "document_title": record.title,
                "chunk_index": chunk_index,
            }
            chunks.append(Document(page_content=text, metadata=metadata))

    return chunks
