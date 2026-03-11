from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document as LangChainDocument


@dataclass
class DocumentRecord:
    source_id: str
    source_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_langchain_document(self) -> LangChainDocument:
        payload = {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            **self.metadata,
        }
        return LangChainDocument(page_content=self.content, metadata=payload)


@dataclass
class DocumentChunk:
    source_id: str
    source_type: str
    title: str
    chunk_index: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_langchain_document(self) -> LangChainDocument:
        payload = {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "title": self.title,
            "document_title": self.title,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }
        return LangChainDocument(page_content=self.content, metadata=payload)


@dataclass
class WebSearchResult:
    source_id: str
    source_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    url: str = ""
    snippet: str = ""


@dataclass
class AnswerSource:
    source_id: str
    source_type: str
    title: str
    citation: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


IngestedDocument = DocumentRecord


def build_document(
    *,
    source_id: str,
    source_type: str,
    title: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        source_id=source_id,
        source_type=source_type,
        title=title,
        content=content,
        metadata=metadata or {},
    )


def build_chunk(
    *,
    source_id: str,
    source_type: str,
    title: str,
    chunk_index: int,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        source_id=source_id,
        source_type=source_type,
        title=title,
        chunk_index=chunk_index,
        content=content,
        metadata=metadata or {},
    )


def build_web_result(
    *,
    source_id: str,
    title: str,
    snippet: str,
    url: str,
    metadata: dict[str, Any] | None = None,
) -> WebSearchResult:
    return WebSearchResult(
        source_id=source_id,
        source_type="web",
        title=title,
        content=snippet,
        snippet=snippet,
        url=url,
        metadata=metadata or {},
    )


def build_answer_source(
    *,
    source_id: str,
    source_type: str,
    title: str,
    citation: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> AnswerSource:
    return AnswerSource(
        source_id=source_id,
        source_type=source_type,
        title=title,
        citation=citation,
        content=content,
        metadata=metadata or {},
    )
