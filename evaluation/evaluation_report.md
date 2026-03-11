# Evaluation Report

## Scope

This report reviews the Hybrid Multi-Document RAG Search Engine against three scenario types:

- static document-grounded queries
- live web queries
- hybrid reasoning queries that mix internal and external evidence

## Strengths

- Multi-document retrieval works across PDFs, text-like files, and optional Wikipedia pages through one normalized ingestion flow.
- FAISS retrieval is fast enough for interactive use and provides transparent chunk-level similarity scores.
- Cross-encoder reranking improves the quality of final document evidence before answer generation.
- Query routing keeps current-events questions from relying only on stale internal documents.
- The UI separates answer, document evidence, and web evidence, which makes source inspection easier.
- Top-document summaries help users scan the strongest local sources before reading raw chunks.

## Scenario assessment

### Static knowledge queries

- Expected behavior: answer should rely mostly on indexed local documents and cite chunked document evidence.
- Current result: strong when the indexed material is relevant and clean, especially for conceptual or explanatory questions.
- Risk: thin or noisy source documents can still lead to incomplete answers.

### Real-time factual queries

- Expected behavior: route to Tavily, treat results as temporary evidence, and keep citations clearly marked as web sources.
- Current result: good coverage for current topics when `TAVILY_API_KEY` is configured.
- Risk: answer quality depends on upstream web snippets and the query router correctly identifying time-sensitive questions.

### Hybrid reasoning queries

- Expected behavior: combine internal chunks with current external snippets under a balanced context budget.
- Current result: useful for comparison-style questions where the internal documents provide background and Tavily adds recency.
- Risk: hybrid answers can still miss nuance if either side of the evidence pool is weak.

## Quality assessment

### Retrieval relevance

- FAISS plus reranking gives a solid baseline for semantic relevance.
- Retrieval quality improves noticeably after cleaning and chunk overlap.

### Answer grounding

- The answer generator is instructed to stay within retrieved context.
- Grounding is generally good, but there is still no hard post-check that verifies every claim against evidence.

### Web vs document separation

- Document and web evidence are displayed separately in the UI.
- Route indicators make it clear whether the answer came from document, web, or hybrid retrieval.

### Citation clarity

- Citation formatting is readable and distinguishes document chunks from Tavily results.
- Citation presence still depends partly on answer-generation behavior, so stricter post-processing would improve reliability.

## Limitations

- Query routing is heuristic-based and may misclassify ambiguous prompts.
- FAISS scores are helpful ranking signals, not calibrated confidence scores.
- Wikipedia ingestion is supported, but content quality still depends on the upstream page.
- The current Groq answer path is stable, but it is a pragmatic provider-specific layer rather than pure LangChain orchestration.
- There is no automated regression harness yet for repeated end-to-end evaluation.

## Future improvements

- Add automated scenario scoring for retrieval precision, citation coverage, and grounding quality.
- Add citation validation that maps answer claims back to the evidence set.
- Introduce BM25 or hybrid sparse-dense retrieval for stronger lexical matching.
- Add knowledge-graph or entity-linking support for structured reasoning across sources.
- Add persistent user sessions and exportable conversation history for longer workflows.
