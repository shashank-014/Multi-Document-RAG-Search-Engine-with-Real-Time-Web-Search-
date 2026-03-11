# Design Decisions

## Why a modular layout

The project is split by pipeline stage so ingestion, indexing, retrieval, web search, RAG assembly, and UI can evolve independently without turning the app into one large script.

## Why a unified schema

Local files, Wikipedia pages, retrieval chunks, and web snippets all move through the same pipeline. Separate but aligned models make it easier to preserve source traceability without forcing every module to reinvent metadata rules.

## Why FAISS

FAISS is a good fit for local semantic retrieval because it is fast, widely used, and simple to persist for a Streamlit-based workflow.

## Why query routing

Not every question needs live web search. A router keeps costs down, avoids noisy external context, and makes it easier to explain why the system picked a specific retrieval path.

## Why keep web search separate

Tavily access sits behind its own module so external retrieval can be tested, swapped, or rate-limited without tangling the internal document pipeline.

## Why keep reranking optional

Cross-encoder reranking improves relevance but adds latency. Keeping it as a dedicated module makes it easy to tune or disable later if response time matters more than marginal retrieval gains.

## Why noisy text hurts retrieval

Embeddings work best when the text reflects real meaning instead of extraction junk. Repeated headers, broken line wraps, and OCR artifacts add fake patterns to the vector space, which lowers semantic match quality and makes retrieval less reliable.

## Why session memory

The deployed app needed a memory approach that works cleanly with Streamlit Cloud and avoids removed LangChain modules. Streamlit session state keeps follow-up questions useful without adding another persistence dependency.

## Why use the direct Groq client

The original plan used LangChain chat wrappers end to end, but production deployment exposed provider-specific request issues. The direct Groq client keeps the answer-generation step stable while the rest of the pipeline still uses LangChain for loaders, chunking, and vector retrieval.

## Security note

Secrets must be read with `st.secrets["KEY_NAME"]`. No API keys should be hardcoded in the repo.
