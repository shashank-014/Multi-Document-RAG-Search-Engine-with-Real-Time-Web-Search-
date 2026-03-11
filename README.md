# Hybrid Multi-Document RAG Search Engine with Real-Time Web Search

## Overview

This project is a production-style hybrid Retrieval-Augmented Generation application that answers questions from indexed internal documents and live web results in one Streamlit chat experience. It combines local semantic search with Tavily-powered web search, then assembles grounded context for Groq-based answer generation with visible evidence tabs, summaries, and citations.

## Core Capabilities

- Multi-document ingestion for PDF, TXT, Markdown, and Wikipedia sources
- Text cleaning and normalization before indexing
- Recursive chunking with overlap for retrieval quality
- FAISS vector indexing with `sentence-transformers/all-MiniLM-L6-v2`
- Semantic similarity search across all indexed documents
- Rule-based query routing for document, web, and hybrid questions
- Tavily real-time web search for fresh external information
- Hybrid context assembly with document and web balancing
- Groq Llama 3 answer generation grounded in retrieved evidence
- Streamlit chatbot UI with route indicators and evidence tabs

## Advanced Features

- Session-based conversation memory for follow-up questions
- Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Lightweight query rewriting for short or vague prompts
- Top-document summaries with retrieval score visibility
- Context deduping before answer generation
- Deterministic source list attachment for safer citation display
- Indexed-source persistence for sidebar recovery after app restarts

## Technology Stack

- Python
- Streamlit
- LangChain
- LangChain Community
- LangChain Text Splitters
- FAISS
- Sentence Transformers
- Tavily Search
- Groq API

## Architecture

High-level pipeline:

`document ingestion -> cleaning -> chunking -> FAISS indexing -> query routing -> semantic retrieval / Tavily search -> reranking -> context assembly -> Groq answer generation -> Streamlit UI`

Source handling:
- Local documents are loaded from `data/documents/`
- Wikipedia topics can be added from the UI and indexed alongside local files
- Tavily results are treated as temporary web evidence and are never persisted into the FAISS index

Answer transparency:
- Query-type indicators show whether a response is document, web, or hybrid driven
- Document and web evidence are separated into dedicated tabs
- FAISS score and rerank score are shown for document evidence
- Top documents are summarized before raw chunk evidence

For a fuller breakdown, see [docs/architecture.md](/Users/SHASHANK/Projects/almax/1/Hybrid-RAG-Search/docs/architecture.md) and [docs/design_decisions.md](/Users/SHASHANK/Projects/almax/1/Hybrid-RAG-Search/docs/design_decisions.md).

## Repository Structure

```text
hybrid-rag-search/
|-- app.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- ingestion/
|   |-- schema.py
|   |-- loaders.py
|   |-- cleaner.py
|-- indexing/
|   |-- chunking.py
|   |-- vector_store.py
|-- retrieval/
|   |-- semantic_search.py
|   |-- reranker.py
|   |-- query_router.py
|   |-- query_rewriter.py
|-- web/
|   |-- tavily_search.py
|-- rag/
|   |-- context_builder.py
|   |-- answer_generator.py
|   |-- citation_formatter.py
|   |-- memory.py
|   |-- summarizer.py
|-- ui/
|   |-- streamlit_ui.py
|   |-- style.css
|-- evaluation/
|   |-- test_queries.py
|   |-- evaluation_report.md
|-- docs/
|   |-- architecture.md
|   |-- design_decisions.md
|-- data/
|   |-- documents/
|       |-- sample_document.txt
|-- faiss_index/
|   |-- index_placeholder.txt
|   |-- source_manifest.json  # created after indexing
```

## Installation

1. Create and activate a Python 3 environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Required Secrets

Configure the following keys in Streamlit secrets:

- `GROQ_API_KEY`
- `TAVILY_API_KEY`

Example for `.streamlit/secrets.toml` in local development:

```toml
GROQ_API_KEY = "your-groq-key"
TAVILY_API_KEY = "your-tavily-key"
```

The application reads secrets with Streamlit native secret handling and never expects `.env` files.

## Running the App

Start the application from the repository root:

```bash
streamlit run app.py
```

## How to Use

1. Upload PDF, TXT, or Markdown files from the sidebar.
2. Optionally add Wikipedia topics, one per line.
3. Click `Index Sources`.
4. Ask a question in the chat input.
5. Inspect the `Answer`, `Document Evidence`, and `Web Evidence` tabs.

## Example Queries

Document queries:
- `What do the uploaded transformer notes say about attention?`
- `What is the role of the Supreme Court in protecting fundamental rights?`

Web queries:
- `What are the latest developments in retrieval augmented generation systems?`
- `What is the latest news about Groq and open-weight LLM serving?`

Hybrid queries:
- `How do the uploaded transformer notes compare with current RAG tooling trends?`
- `Compare the indexed court material with current legal commentary on fundamental rights.`

## Deployment on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Point the app at `app.py`.
4. Add `GROQ_API_KEY` and `TAVILY_API_KEY` in the app's Secrets section.
5. Deploy and index sources from the sidebar after the app starts.

Notes:
- `data/documents/` and `faiss_index/` include visible placeholder files so the folders appear on GitHub.
- The FAISS index is created locally inside `faiss_index/` after indexing.
- A small source manifest is also written there so indexed file names and Wikipedia topics can be shown again after a restart.

## Evaluation

The repository includes:
- [evaluation/test_queries.py](/Users/SHASHANK/Projects/almax/1/Hybrid-RAG-Search/evaluation/test_queries.py) for scenario-based prompts
- [evaluation/evaluation_report.md](/Users/SHASHANK/Projects/almax/1/Hybrid-RAG-Search/evaluation/evaluation_report.md) for strengths, limitations, and future improvements

## Future Improvements

- Add automated end-to-end evaluation runs with citation scoring
- Add BM25 plus dense retrieval for stronger lexical recall
- Add stricter citation validation against evidence blocks
- Add exportable conversations and persistent user sessions
- Add richer answer highlighting mapped back to source snippets
