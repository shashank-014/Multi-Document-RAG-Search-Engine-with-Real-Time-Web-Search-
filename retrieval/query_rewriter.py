import re

SHORT_QUERY_TERMS = {"rag": "retrieval augmented generation", "llm": "large language model"}
VAGUE_TERMS = {"latest", "recent", "research", "news", "update", "updates", "new"}



def _expand_terms(text: str) -> str:
    expanded = text
    for short_term, full_term in SHORT_QUERY_TERMS.items():
        expanded = re.sub(rf"\b{short_term}\b", full_term, expanded, flags=re.IGNORECASE)
    return expanded



def _needs_rewrite(query: str) -> bool:
    words = query.split()
    return len(words) <= 4 or any(term in query.lower() for term in VAGUE_TERMS)



def _build_rewrite(query: str) -> str:
    expanded = _expand_terms(query)
    cleaned = re.sub(r"\s+", " ", expanded).strip()

    if "latest" in cleaned.lower() and "research" in cleaned.lower():
        rewritten = cleaned.lower().replace("latest research", "latest research developments")
        return rewritten.replace(
            "retrieval augmented generation",
            "retrieval augmented generation systems",
        )

    if len(cleaned.split()) <= 4:
        return f"detailed information about {cleaned}"

    return cleaned



def rewrite_query(query: str) -> dict[str, str | bool]:
    cleaned = re.sub(r"\s+", " ", query).strip()
    rewritten_query = _build_rewrite(cleaned) if _needs_rewrite(cleaned) else cleaned
    web_query = re.sub(
        r"\b(pdf|document|internal|uploaded)\b",
        "",
        rewritten_query,
        flags=re.IGNORECASE,
    )
    web_query = re.sub(r"\s+", " ", web_query).strip()

    return {
        "original_query": cleaned,
        "vector_query": rewritten_query,
        "web_query": web_query or rewritten_query,
        "rewritten_query": rewritten_query,
        "was_rewritten": rewritten_query != cleaned,
    }
