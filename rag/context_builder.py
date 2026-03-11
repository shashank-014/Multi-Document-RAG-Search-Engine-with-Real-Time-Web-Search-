from langchain_core.documents import Document

from config import HYBRID_DOC_LIMIT, HYBRID_WEB_LIMIT, MAX_CONTEXT_CHARS
from ingestion.schema import build_answer_source
from rag.citation_formatter import format_doc_citation, format_web_citation


def _trim_block(block: str, remaining: int) -> str:
    if len(block) <= remaining:
        return block
    return block[: max(remaining - 3, 0)] + "..."


def _doc_key(doc: Document) -> tuple[str, object, str]:
    return (
        str(doc.metadata.get("source_id", doc.metadata.get("title", ""))),
        doc.metadata.get("chunk_index", -1),
        " ".join(doc.page_content.split())[:180],
    )


def _web_key(result: dict[str, str]) -> tuple[str, str]:
    return (
        result.get("url", ""),
        " ".join(result.get("snippet", "").split())[:180],
    )


def _dedupe_docs(doc_chunks: list[Document]) -> list[Document]:
    seen: set[tuple[str, object, str]] = set()
    deduped: list[Document] = []
    for doc in doc_chunks:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def _dedupe_web(web_results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for result in web_results:
        key = _web_key(result)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _select_sources(
    doc_chunks: list[Document],
    web_results: list[dict[str, str]],
    query_type: str,
) -> tuple[list[Document], list[dict[str, str]]]:
    if query_type == "hybrid":
        return doc_chunks[:HYBRID_DOC_LIMIT], web_results[:HYBRID_WEB_LIMIT]
    if query_type == "web":
        return [], web_results
    return doc_chunks, []


def build_context(
    doc_chunks: list[Document],
    web_results: list[dict[str, str]],
    query_type: str,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> dict[str, object]:
    selected_docs, selected_web = _select_sources(_dedupe_docs(doc_chunks), _dedupe_web(web_results), query_type)
    blocks: list[str] = []
    citations: list[str] = []
    answer_sources = []
    used = 0

    doc_evidence = []
    for doc in selected_docs:
        citation = format_doc_citation(doc)
        block = f"{citation}\n{doc.page_content.strip()}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        block = _trim_block(block, remaining)
        blocks.append(block)
        citations.append(citation)
        used += len(block) + 2
        answer_sources.append(
            build_answer_source(
                source_id=str(doc.metadata.get("source_id", doc.metadata.get("title", "document"))),
                source_type=str(doc.metadata.get("source_type", "document")),
                title=str(doc.metadata.get("document_title", doc.metadata.get("title", "Unknown document"))),
                citation=citation,
                content=doc.page_content,
                metadata=dict(doc.metadata),
            )
        )
        doc_evidence.append(
            {
                "citation": citation,
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    web_evidence = []
    for result in selected_web:
        citation = format_web_citation(result)
        block = f"{citation}\n{result.get('snippet', '').strip()}\nURL: {result.get('url', '')}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        block = _trim_block(block, remaining)
        blocks.append(block)
        citations.append(citation)
        used += len(block) + 2
        answer_sources.append(
            build_answer_source(
                source_id=str(result.get("source_id", result.get("url", result.get("title", "web")))),
                source_type="web",
                title=str(result.get("title", "Untitled web result")),
                citation=citation,
                content=result.get("snippet", ""),
                metadata={"url": result.get("url", "")},
            )
        )
        web_evidence.append(
            {
                "citation": citation,
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "url": result.get("url", ""),
            }
        )

    unique_citations = list(dict.fromkeys(citations))
    return {
        "context": "\n\n".join(blocks),
        "doc_evidence": doc_evidence,
        "web_evidence": web_evidence,
        "citations": unique_citations,
        "answer_sources": answer_sources,
    }
