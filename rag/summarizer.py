from collections import OrderedDict

from langchain_core.documents import Document



def _clean_snippet(text: str, limit: int = 220) -> str:
    snippet = " ".join(text.split())
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3].rstrip() + "..."



def summarize_documents(docs: list[Document], max_items: int = 3) -> list[dict[str, object]]:
    summaries: OrderedDict[str, dict[str, object]] = OrderedDict()

    for doc in docs:
        title = doc.metadata.get("document_title") or doc.metadata.get("title") or "Unknown document"
        if title in summaries:
            continue
        summaries[title] = {
            "title": title,
            "summary": _clean_snippet(doc.page_content),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "similarity_score": doc.metadata.get("similarity_score"),
            "rerank_score": doc.metadata.get("rerank_score"),
        }
        if len(summaries) >= max_items:
            break

    return list(summaries.values())
