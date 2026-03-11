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

ROUTE_STYLES = {
    "document": "route-doc",
    "web": "route-web",
    "hybrid": "route-hybrid",
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


def _active_files_on_disk(file_names: list[str]) -> list[str]:
    active: list[str] = []
    for name in file_names:
        if (DATA_DIR / name).exists():
            active.append(name)
    return sorted(dict.fromkeys(active))


def _sync_active_index_state() -> None:
    manifest = _load_source_manifest()
    active_files = _active_files_on_disk(manifest["files"])
    wiki_topics = sorted(dict.fromkeys(manifest["wiki_topics"]))

    st.session_state.uploaded_files = active_files
    st.session_state.indexed_wiki_topics = wiki_topics
    _save_source_manifest(active_files, wiki_topics)

    if not active_files and not wiki_topics:
        st.session_state.vector_store = None


def _ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    create_memory()
    _sync_active_index_state()


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


def _get_indexed_sources() -> dict[str, list[str]]:
    return {
        "files": sorted(dict.fromkeys(st.session_state.get("uploaded_files", []))),
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
    if not _has_indexed_sources():
        st.session_state.vector_store = None
        return None

    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
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


def _route_badge(route: str) -> str:
    css_class = ROUTE_STYLES.get(route, "route-doc")
    label = ROUTE_LABELS.get(route, route)
    return f'<span class="route-badge {css_class}">{label}</span>'


def _metric_card(label: str, value: str, tone: str = "") -> str:
    css_class = f"metric-card {tone}".strip()
    return f'<div class="{css_class}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'


def _format_source_name(name: str, limit: int = 34) -> str:
    if len(name) <= limit:
        return name
    return name[: limit - 3] + "..."


def _split_answer_and_sources(answer_text: str) -> tuple[str, list[str]]:
    if "Sources:" not in answer_text:
        return answer_text.strip(), []

    body, source_block = answer_text.split("Sources:", maxsplit=1)
    sources = [item.strip() for item in source_block.split(";") if item.strip()]
    return body.strip(), sources


def _render_nav_links() -> None:
    st.markdown(
        '<div class="page-nav"><a href="#top-anchor">Back to top</a><a href="#latest-anchor">Jump to latest</a></div>',
        unsafe_allow_html=True,
    )


def _render_sidebar() -> tuple[list, list[str], bool, bool, bool]:
    indexed_sources = _get_indexed_sources()

    with st.sidebar:
        st.markdown('<div class="sidebar-section sidebar-hero">', unsafe_allow_html=True)
        st.markdown("### Hybrid RAG Copilot")
        st.caption("Search across indexed documents, Wikipedia topics, and live Tavily results from one chat interface.")
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
            help="Optional encyclopedia sources that will be added to the active index.",
        )
        use_web = st.toggle("Enable Web Search", value=True)
        st.caption("Each indexing run replaces the active index with the sources selected above.")
        index_clicked = st.button("Index Sources", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Active Index")
        if indexed_sources["files"]:
            st.markdown("**Files currently in use**")
            for source in indexed_sources["files"]:
                st.markdown(f'<div class="source-pill">{_format_source_name(source)}</div>', unsafe_allow_html=True)
        if indexed_sources["wiki_topics"]:
            st.markdown("**Wikipedia currently in use**")
            for topic in indexed_sources["wiki_topics"]:
                st.markdown(f'<div class="source-pill source-pill-wiki">{_format_source_name(topic)}</div>', unsafe_allow_html=True)
        if not indexed_sources["files"] and not indexed_sources["wiki_topics"]:
            st.caption("No active index yet")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Available in data/documents", expanded=False):
            active_files = indexed_sources["files"]
            if active_files:
                st.caption("Only files currently used by the active index are shown here.")
                for source in active_files:
                    st.write(f"- {source}")
            else:
                st.caption("No active document files in use.")

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Controls")
        clear_chat = st.button("Clear Chat History", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    return uploaded_files or [], _parse_wiki_topics(wiki_topics_raw), use_web, index_clicked, clear_chat


def _render_empty_state(use_web: bool) -> None:
    web_text = "Enabled" if use_web else "Disabled"
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.title(PAGE_TITLE)
    st.caption(PAGE_SUBTITLE)
    metrics = st.columns(3)
    with metrics[0]:
        st.markdown(_metric_card("Active files", str(len(_get_indexed_sources()["files"])), "metric-soft"), unsafe_allow_html=True)
    with metrics[1]:
        st.markdown(_metric_card("Wikipedia topics", str(len(_get_indexed_sources()["wiki_topics"])), "metric-warm"), unsafe_allow_html=True)
    with metrics[2]:
        st.markdown(_metric_card("Web search", web_text, "metric-cool"), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    tips = st.columns(2)
    with tips[0]:
        st.markdown('<div class="card helper-card">', unsafe_allow_html=True)
        st.markdown("### Ask Better Questions")
        st.write("Use document questions for internal material, web questions for current events, and hybrid questions when you want both.")
        st.markdown("</div>", unsafe_allow_html=True)
    with tips[1]:
        st.markdown('<div class="card helper-card">', unsafe_allow_html=True)
        st.markdown("### What You Can Index")
        st.write("PDF, TXT, Markdown, and optional Wikipedia topics can all feed the active index.")
        st.markdown("</div>", unsafe_allow_html=True)


def _render_chat_history() -> None:
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        bubble_class = "chat-user" if message["role"] == "user" else "chat-ai"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(
                f'<div class="{bubble_class}">{message["content"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_summary_cards(summaries: list[dict[str, object]]) -> None:
    if not summaries:
        return

    st.markdown("### Top Documents")
    for item in summaries:
        score = item.get("similarity_score")
        rerank = item.get("rerank_score")
        score_text = f"FAISS: {score:.4f}" if isinstance(score, float) else "FAISS: n/a"
        rerank_text = f"Rerank: {rerank:.4f}" if isinstance(rerank, float) else "Rerank: n/a"
        st.markdown('<div class="evidence-card summary-card">', unsafe_allow_html=True)
        st.markdown(f"**{item['title']}**")
        st.caption(f"{score_text} | {rerank_text}")
        st.write(item["summary"])
        st.markdown("</div>", unsafe_allow_html=True)


def _render_doc_evidence(doc_evidence: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    _render_summary_cards(summaries)

    if summaries and doc_evidence:
        st.divider()

    if not doc_evidence:
        st.caption("No document evidence used.")
        return

    st.markdown("### Retrieved Chunks")
    for index, item in enumerate(doc_evidence, start=1):
        meta = item["metadata"]
        score = meta.get("similarity_score")
        rerank = meta.get("rerank_score")
        score_text = f"{score:.4f}" if isinstance(score, float) else "n/a"
        rerank_text = f"{rerank:.4f}" if isinstance(rerank, float) else "n/a"
        with st.expander(f"{index}. {item['citation']}", expanded=index == 1):
            st.caption(
                f"Title: {meta.get('document_title', 'Unknown')} | "
                f"Chunk: {meta.get('chunk_index', 'n/a')} | "
                f"FAISS score: {score_text} | "
                f"Rerank score: {rerank_text}"
            )
            st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
            st.write(item["content"])
            st.markdown("</div>", unsafe_allow_html=True)


def _render_web_evidence(web_evidence: list[dict[str, object]]) -> None:
    if not web_evidence:
        st.caption("No web evidence used.")
        return

    st.markdown("### Tavily Results")
    for index, item in enumerate(web_evidence, start=1):
        with st.expander(f"{index}. {item['citation']}", expanded=index == 1):
            st.markdown('<div class="evidence-card evidence-web">', unsafe_allow_html=True)
            st.write(item["snippet"])
            if item["url"]:
                st.markdown(f"[Open source]({item['url']})")
            st.markdown("</div>", unsafe_allow_html=True)


def _render_answer_tab(answer_text: str, route: str, summaries: list[dict[str, object]]) -> None:
    body, sources = _split_answer_and_sources(answer_text)
    st.markdown(_route_badge(route), unsafe_allow_html=True)
    st.markdown('<div class="answer-panel">', unsafe_allow_html=True)
    st.write(body or "No answer generated.")
    st.markdown("</div>", unsafe_allow_html=True)

    if sources:
        st.markdown("### Sources")
        st.markdown('<div class="source-chip-row">', unsafe_allow_html=True)
        for source in sources:
            st.markdown(f'<span class="answer-source-chip">{source}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if summaries:
        st.markdown("### Quick Source Read")
        for item in summaries[:2]:
            st.markdown('<div class="mini-summary-card">', unsafe_allow_html=True)
            st.markdown(f"**{item['title']}**")
            st.write(item["summary"])
            st.markdown("</div>", unsafe_allow_html=True)


def _handle_indexing(uploaded_files: list, wiki_topics: list[str]) -> None:
    if not uploaded_files and not wiki_topics:
        st.warning("Please upload documents or add Wikipedia topics before indexing.")
        return

    saved_paths = _save_uploaded_files(uploaded_files)
    active_files = [path.name for path in saved_paths]
    active_topics = list(dict.fromkeys(wiki_topics))

    chunk_count = _index_sources(saved_paths, active_topics)
    if chunk_count:
        st.session_state.uploaded_files = active_files
        st.session_state.indexed_wiki_topics = active_topics
        _save_source_manifest(active_files, active_topics)
        st.success(f"Indexed {chunk_count} chunks across the selected sources.")
    else:
        st.warning("No sources were indexed. Please upload valid files or check the Wikipedia topic names.")


def run_app() -> None:
    ensure_dirs()
    create_memory()
    _ensure_state()
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    _load_css()
    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

    uploaded_files, wiki_topics, use_web, index_clicked, clear_chat = _render_sidebar()

    if clear_chat:
        _reset_chat()
        st.success("Chat history cleared.")

    if index_clicked:
        _handle_indexing(uploaded_files, wiki_topics)

    _render_nav_links()

    if not st.session_state.messages:
        _render_empty_state(use_web)
    else:
        header_left, header_right = st.columns([3, 2])
        with header_left:
            st.markdown('<div class="compact-hero">', unsafe_allow_html=True)
            st.markdown(f"<div class='hero-title'>{PAGE_TITLE}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='hero-subtitle'>{PAGE_SUBTITLE}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with header_right:
            indexed_sources = _get_indexed_sources()
            cols = st.columns(3)
            with cols[0]:
                st.markdown(_metric_card("Files", str(len(indexed_sources["files"])), "metric-soft"), unsafe_allow_html=True)
            with cols[1]:
                st.markdown(_metric_card("Wiki", str(len(indexed_sources["wiki_topics"])), "metric-warm"), unsafe_allow_html=True)
            with cols[2]:
                st.markdown(_metric_card("Web", "On" if use_web else "Off", "metric-cool"), unsafe_allow_html=True)

    chat_panel = st.container()
    with chat_panel:
        _render_chat_history()

    st.markdown('<div id="latest-anchor"></div>', unsafe_allow_html=True)
    query = st.chat_input("Ask a question about your active sources, Wikipedia topics, or current events...")
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
        st.markdown(_route_badge(result["route"]), unsafe_allow_html=True)
        if result["rewritten"]["was_rewritten"]:
            st.caption(f"Rewritten query: {result['rewritten']['rewritten_query']}")
        for notice in result["notices"]:
            st.warning(notice)

        thinking = st.empty()
        live_answer = st.empty()
        thinking.markdown('<div class="thinking-pill">Thinking...</div>', unsafe_allow_html=True)
        answer_text = ""
        for chunk in stream_answer(query, result["context"], result["memory_text"], citations=result["citations"]):
            answer_text += chunk
            preview_text, _ = _split_answer_and_sources(answer_text)
            live_answer.markdown(f'<div class="chat-ai">{preview_text}</div>', unsafe_allow_html=True)
        thinking.empty()
        live_answer.empty()

        answer_tab, doc_tab, web_tab = st.tabs(["Answer", "Document Evidence", "Web Evidence"])
        with answer_tab:
            _render_answer_tab(answer_text, result["route"], result["summaries"])
        with doc_tab:
            _render_doc_evidence(result["doc_evidence"], result["summaries"])
        with web_tab:
            _render_web_evidence(result["web_evidence"])

    body_text, _ = _split_answer_and_sources(answer_text)
    save_turn(query, body_text)
    st.session_state.messages.append({"role": "assistant", "content": body_text})
