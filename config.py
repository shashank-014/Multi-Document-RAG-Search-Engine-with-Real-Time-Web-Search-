from pathlib import Path
from typing import Any, Optional

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "documents"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
DELETED_DIR = BASE_DIR / "DELETED_FILES"
SOURCE_MANIFEST_PATH = FAISS_INDEX_DIR / "source_manifest.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5
HYBRID_DOC_LIMIT = 3
HYBRID_WEB_LIMIT = 2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_CONTEXT_CHARS = 6000


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DELETED_DIR.mkdir(parents=True, exist_ok=True)


def get_secret(key_name: str, default: Optional[Any] = None) -> Optional[str]:
    try:
        return st.secrets[key_name]
    except Exception:
        return default
