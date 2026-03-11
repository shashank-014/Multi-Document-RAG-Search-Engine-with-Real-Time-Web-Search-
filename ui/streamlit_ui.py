import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import DATA_DIR, DELETED_DIR, SOURCE_MANIFEST_PATH, ensure_dirs, get_secret
from indexing.chunking import chunk_documents
from indexing.vector_store import index_documents, load_faiss_index
from ingestion.loaders import load_sources
from rag.answer_generator import stream_answer
from rag.context_builder import build_context
from rag.memory import create_memory, load_memory_text, save_turn
from rag.summarizer import summarize_documents
from retrieval.query_rewriter import rewrite_query
from retrieval.query_router import route_query
from retrieval.reranker import rerank_documents
from retrieval.semantic_search import search_documents
from web.tavily_search import search_web

ROUTE_LABELS = {
    "document": "📄 Document answer",
    "web": "🌐 Web answer",
    "hybrid": "🔀 Hybrid answer",
}

PAGE_TITLE = "Hybrid Multi-Document RAG Search Engine"
PAGE_SUBTITLE = "Ask questions across internal documents and real-time web search"


def _load_css() -> None:
    with open("ui/style.css", encoding="utf-8") as file_handle:
        st.markdown(f"<style>{file_handle.read()}</style>", unsafe_allow_html=True)


def _archive_existing(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y-%m-%d")
    archived = DELETED_DIR / f"data_documents_{stamp}_prev_{path.name}"
    path.replace(archived)


def _load_source_manifest() -> dict[str, list[str]]:
    if not SOURCE_MANIFEST_PATH.exists():
        return {"files": [], "wiki_topics": []}

    try:
        data = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"files": [], "wiki_topics": []}

    return {
        "files": data.get("files", []),
        "wiki_topics": data.get("wiki_topics", []),
    }


def _save_source_manifest(files: list[str], wiki_topics: list[str]) -> None:
    ensure_dirs()
    payload = {
        "files": sorted(dict.fromkeys(files)),
        "wiki_topics": sorted(dict.fromkeys(wiki_topics)),
    }
    SOURCE_MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_state() -> None:
    manifest = _load_source_manifest()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = manifest["files"]
    if "indexed_wiki_topics" not in st.session_state:
        st.session_state.indexed_wiki_topics = manifest["wiki_topics"]
    create_memory()


def _save_uploaded_files(uploaded_files: list) -> list[Path]:
    ensure_dirs()
    saved_paths: list[Path] = []

    for uploaded in uploaded_files:
        target = DATA_DIR / uploaded.name
        _archive_existing(target)
        target.write_bytes(uploaded.getbuffer())
        saved_paths.append(target)

    return saved_paths


def _reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.chat_history = []


def _parse_wiki_topics(raw_topics: str) -> list[str]:
    return [line.strip() for line in raw_topics.splitlines() if line.strip()]


def _get_indexed_titles() -> list[str]:
    ignored = {".keep", "sample_document.txt", "source_manifest.json"}
    disk_files = [path.name for path in DATA_DIR.glob("*") if path.is_file() and path.name not in ignored]
    manifest_files = st.session_state.get("uploaded_files", [])
    return sorted(dict.fromkeys(disk_files + manifest_files))


def _get_indexed_sources() -> dict[str, list[str]]:
    return {
        "files": _get_indexed_titles(),
        "wiki_topics": sorted(dict.fromkeys(st.session_state.get("indexed_wiki_topics", []))),
    }


def _has_indexed_sources() -> bool:
    sources = _get_indexed_sources()
    return bool(sources["files"] or sources["wiki_topics"])


def _index_sources(file_paths: list[Path], wiki_topics: list[str]) -> int:
    records = load_sources(file_paths, wiki_topics=wiki_topics)
    if not records:
        return 0

    st.session_state.ingested_records = records
    chunks = chunk_documents(records)
    store = index_documents(chunks)
    st.session_state.vector_store = store
    return len(chunks)


def _load_store_from_disk():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_faiss_index()
    return st.session_state.vector_store


def _build_notices(route: str, use_web: bool, store) -> list[str]:
    notices: list[str] = []
    if not _has_indexed_sources() and route in {"document", "hybrid"}:
        notices.append("No indexed documents are available yet. Upload files or add Wikipedia topics, then run indexing.")
    if store is None and route in {"document", "hybrid"}:
        notices.append("Please upload documents to create the vector index.")
    if use_web and route in {"web", "hybrid"} and not get_secret("TAVILY_API_KEY"):
        notices.append("TAVILY_API_KEY is missing in Streamlit secrets. Disable web search or add the key.")
    if not get_secret("GROQ_API_KEY"):
        notices.append("GROQ_API_KEY is missing in Streamlit secrets.")
    return notices


def _run_query(query: str, use_web: bool) -> dict[str, object]:
    store = _load_store_from_disk()
    route = route_query(query)
    rewritten = rewrite_query(query)
    notices = _build_notices(route, use_web, store)

    doc_hits = []
    if store and route in {"document", "hybrid"}:
        doc_hits = search_documents(rewritten["vector_query"], store)
        doc_hits = rerank_documents(rewritten["vector_query"], doc_hits, top_k=5)

    web_hits = []
    if use_web and route in {"web", "hybrid"} and get_secret("TAVILY_API_KEY"):
        web_hits = search_web(rewritten["web_query"], top_k=5)

    payload = build_context(doc_hits, web_hits, route)
    summaries = summarize_documents(doc_hits, max_items=3)

    return {
        "route": route,
        "route_label": ROUTE_LABELS.get(route, route),
        "rewritten": rewritten,
        "doc_evidence": payload["doc_evidence"],
        "web_evidence": payload["web_evidence"],
        "context": payload["context"],
        "citations": payload["citations"],
        "summaries": summaries,
        "memory_text": load_memory_text(),
        "notices": notices,
    }


def _render_sidebar() -> tuple[list, list[str], bool, bool, bool]:
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("About")
        st.markdown("**Hybrid Multi-Document RAG Search Engine**")
        st.write("This assistant can:")
        st.write("• Answer questions from uploaded documents")
        st.write("• Search the web using Tavily")
        st.write("• Combine internal evidence with live sources")
        st.write("• Return answers with transparent citations")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("How to Use")
        st.write("1. Upload PDF, TXT, or Markdown files")
        st.write("2. Add Wikipedia topics if needed")
        st.write("3. Click Index Sources")
        st.write("4. Ask questions in the chat")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Document Management")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Add PDF, TXT, or Markdown files for ingestion.",
        )
        st.caption("Drag and drop local source files here.")
        wiki_topics_raw = st.text_area(
            "Wikipedia Topics",
            placeholder="Enter one topic per line",
            help="Optional encyclopedia sources that will be added to the index.",
        )
        use_web = st.toggle("Enable Web Search", value=True)
        index_clicked = st.button("Index Sources", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Indexed Sources")
        indexed_sources = _get_indexed_sources()
        if indexed_sources["files"]:
            st.markdown("**Files**")
            for source in indexed_sources["files"]:
                st.write(f"- {source}")
        if indexed_sources["wiki_topics"]:
            st.markdown("**Wikipedia**")
            for topic in indexed_sources["wiki_topics"]:
                st.write(f"- {topic}")
        if not indexed_sources["files"] and not indexed_sources["wiki_topics"]:
            st.caption("No indexed sources yet")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Controls")
        clear_chat = st.button("Clear Chat History", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    return uploaded_files or [], _parse_wiki_topics(wiki_topics_raw), use_web, index_clicked, clear_chat


def _render_chat_history() -> None:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        bubble_class = "chat-user" if message["role"] == "user" else "chat-ai"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(
                f'<div class="{bubble_class}">{message["content"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_doc_evidence(doc_evidence: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    if summaries:
        st.markdown("### Top Documents")
        for item in summaries:
            score = item.get("similarity_score")
            rerank = item.get("rerank_score")
            score_text = f"FAISS: {score:.4f}" if isinstance(score, float) else "FAISS: n/a"
            rerank_text = f"Rerank: {rerank:.4f}" if isinstance(rerank, float) else "Rerank: n/a"
            st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
            st.markdown(f"**{item['title']}**")
            st.caption(f"{score_text} | {rerank_text}")
            st.write(item["summary"])
            st.markdown("</div>", unsafe_allow_html=True)
        st.divider()

    if not doc_evidence:
        st.caption("No document evidence used.")
        return

    for item in doc_evidence:
        meta = item["metadata"]
        score = meta.get("similarity_score")
        rerank = meta.get("rerank_score")
        score_text = f"{score:.4f}" if isinstance(score, float) else "n/a"
        rerank_text = f"{rerank:.4f}" if isinstance(rerank, float) else "n/a"
        st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
        st.markdown(f"**{item['citation']}**")
        st.caption(
            f"Title: {meta.get('document_title', 'Unknown')} | "
            f"Chunk: {meta.get('chunk_index', 'n/a')} | "
            f"FAISS score: {score_text} | "
            f"Rerank score: {rerank_text}"
        )
        st.write(item["content"])
        st.markdown("</div>", unsafe_allow_html=True)


def _render_web_evidence(web_evidence: list[dict[str, object]]) -> None:
    if not web_evidence:
        st.caption("No web evidence used.")
        return

    for item in web_evidence:
        st.markdown('<div class="evidence-card evidence-web">', unsafe_allow_html=True)
        st.markdown(f"**{item['citation']}**")
        st.write(item["snippet"])
        if item["url"]:
            st.markdown(f"[Open source]({item['url']})")
        st.markdown("</div>", unsafe_allow_html=True)


def _handle_indexing(uploaded_files: list, wiki_topics: list[str]) -> None:
    if not uploaded_files and not wiki_topics:
        st.warning("Please upload documents or add Wikipedia topics before indexing.")
        return

    saved_paths = _save_uploaded_files(uploaded_files)
    existing_names = list(st.session_state.uploaded_files)
    for path in saved_paths:
        if path.name not in existing_names:
            existing_names.append(path.name)
    st.session_state.uploaded_files = sorted(dict.fromkeys(existing_names))

    existing_topics = list(st.session_state.indexed_wiki_topics)
    for topic in wiki_topics:
        if topic not in existing_topics:
            existing_topics.append(topic)
    st.session_state.indexed_wiki_topics = sorted(dict.fromkeys(existing_topics))

    _save_source_manifest(st.session_state.uploaded_files, st.session_state.indexed_wiki_topics)

    chunk_count = _index_sources(saved_paths, wiki_topics)
    if chunk_count:
        st.success(f"Indexed {chunk_count} chunks across local and Wikipedia sources.")
    else:
        st.warning("No sources were indexed. Please upload valid files or check the Wikipedia topic names.")


def run_app() -> None:
    ensure_dirs()
    create_memory()
    _ensure_state()
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    _load_css()

    uploaded_files, wiki_topics, use_web, index_clicked, clear_chat = _render_sidebar()

    if clear_chat:
        _reset_chat()
        st.success("Chat history cleared.")

    if index_clicked:
        _handle_indexing(uploaded_files, wiki_topics)

    header_col, info_col = st.columns([3, 2])
    with header_col:
        st.title(PAGE_TITLE)
        st.caption(PAGE_SUBTITLE)
    with info_col:
        st.markdown('<div class="card status-card">', unsafe_allow_html=True)
        st.markdown("### Search Mode")
        st.write("Use document retrieval for internal knowledge, Tavily for current information, or hybrid mode for both.")
        st.markdown("</div>", unsafe_allow_html=True)

    chat_panel = st.container()
    with chat_panel:
        _render_chat_history()

    query = st.chat_input("Ask a question about your documents, Wikipedia topics, or current events...")
    if not query:
        return

    if not _has_indexed_sources():
        st.warning("Please upload documents or add Wikipedia topics before asking questions.")
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f'<div class="chat-user">{query}</div>', unsafe_allow_html=True)

    result = _run_query(query, use_web)

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(result["route_label"])
        if result["rewritten"]["was_rewritten"]:
            st.caption(f"Rewritten query: {result['rewritten']['rewritten_query']}")
        for notice in result["notices"]:
            st.warning(notice)

        thinking = st.empty()
        bubble = st.empty()
        thinking.markdown("Thinking...")
        answer_text = ""
        for chunk in stream_answer(query, result["context"], result["memory_text"], citations=result["citations"]):
            answer_text += chunk
            bubble.markdown(f'<div class="chat-ai">{answer_text}</div>', unsafe_allow_html=True)
        thinking.empty()

        answer_tab, doc_tab, web_tab = st.tabs(["Answer", "Document Evidence", "Web Evidence"])
        with answer_tab:
            st.markdown(f'<div class="chat-ai">{answer_text}</div>', unsafe_allow_html=True)
        with doc_tab:
            _render_doc_evidence(result["doc_evidence"], result["summaries"])
        with web_tab:
            _render_web_evidence(result["web_evidence"])

    save_turn(query, answer_text)
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
