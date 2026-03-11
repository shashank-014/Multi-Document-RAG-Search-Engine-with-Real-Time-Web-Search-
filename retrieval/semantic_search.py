from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import DEFAULT_TOP_K



def search_documents(query: str, store: FAISS, top_k: int = DEFAULT_TOP_K) -> list[Document]:
    results = store.similarity_search_with_score(query, k=top_k)
    ranked: list[Document] = []

    for rank, (doc, score) in enumerate(results, start=1):
        doc.metadata = {
            **doc.metadata,
            "similarity_score": float(score),
            "retrieval_rank": rank,
        }
        ranked.append(doc)

    return ranked
