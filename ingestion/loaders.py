from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WikipediaLoader

from ingestion.cleaner import clean_text
from ingestion.schema import IngestedDocument, build_document


SUPPORTED_TEXT_EXTS = {".txt", ".md", ".rst"}



def _normalize_doc(doc, source_type: str, source_id: str, title: str) -> IngestedDocument:
    metadata = dict(doc.metadata or {})
    content = clean_text(doc.page_content)
    return build_document(
        source_id=source_id,
        source_type=source_type,
        title=title,
        content=content,
        metadata=metadata,
    )



def load_pdf(file_path: str | Path) -> list[IngestedDocument]:
    path = Path(file_path)
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    return [
        _normalize_doc(doc, "pdf", str(path), path.name)
        for doc in docs
        if doc.page_content.strip()
    ]



def load_text(file_path: str | Path) -> list[IngestedDocument]:
    path = Path(file_path)
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    return [
        _normalize_doc(doc, "text", str(path), path.name)
        for doc in docs
        if doc.page_content.strip()
    ]



def load_wikipedia(topic: str, *, load_max_docs: int = 3) -> list[IngestedDocument]:
    loader = WikipediaLoader(query=topic, load_max_docs=load_max_docs)
    docs = loader.load()
    normalized = []
    for doc in docs:
        title = doc.metadata.get("title", topic)
        source_id = doc.metadata.get("source", title)
        normalized.append(_normalize_doc(doc, "wikipedia", source_id, title))
    return normalized



def load_sources(file_paths: Iterable[str | Path], wiki_topics: Iterable[str] | None = None) -> list[IngestedDocument]:
    records: list[IngestedDocument] = []

    for file_path in file_paths:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            records.extend(load_pdf(path))
            continue
        if path.suffix.lower() in SUPPORTED_TEXT_EXTS:
            records.extend(load_text(path))

    for topic in wiki_topics or []:
        if topic.strip():
            records.extend(load_wikipedia(topic.strip()))

    return records
