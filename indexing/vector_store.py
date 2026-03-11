from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import EMBEDDING_MODEL, FAISS_INDEX_DIR, ensure_dirs



def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)



def index_documents(chunks: list[Document], index_dir: str | Path = FAISS_INDEX_DIR) -> FAISS:
    if not chunks:
        raise ValueError("No chunks provided for indexing.")

    ensure_dirs()
    target_dir = Path(index_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(target_dir))
    return store



def load_faiss_index(index_dir: str | Path = FAISS_INDEX_DIR) -> FAISS | None:
    target_dir = Path(index_dir)
    index_file = target_dir / "index.faiss"

    if not index_file.exists():
        return None

    return FAISS.load_local(
        str(target_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
