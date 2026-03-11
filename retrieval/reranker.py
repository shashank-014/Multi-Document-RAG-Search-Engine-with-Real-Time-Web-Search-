from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import RERANK_MODEL


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)



def rerank_documents(query: str, docs: list[Document], top_k: int | None = None) -> list[Document]:
    if not docs:
        return []

    model = _get_model()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    reranked: list[Document] = []
    for doc, score in zip(docs, scores):
        doc.metadata = {
            **doc.metadata,
            "rerank_score": float(score),
        }
        reranked.append(doc)

    reranked.sort(key=lambda doc: doc.metadata.get("rerank_score", 0.0), reverse=True)

    for rank, doc in enumerate(reranked, start=1):
        doc.metadata["rerank_rank"] = rank

    return reranked[:top_k] if top_k else reranked
