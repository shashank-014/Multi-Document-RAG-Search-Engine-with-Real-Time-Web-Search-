from langchain_core.documents import Document


def _clean_title(title: str) -> str:
    return " ".join((title or "Unknown source").split())


def format_doc_citation(doc: Document) -> str:
    title = _clean_title(doc.metadata.get("document_title") or doc.metadata.get("title") or "Unknown document")
    chunk_index = doc.metadata.get("chunk_index", 0)
    return f"[Doc] {title} - chunk{chunk_index}"


def format_web_citation(result: dict[str, str]) -> str:
    title = _clean_title(result.get("title") or "Untitled web result")
    return f"[Web] Tavily - {title}"
