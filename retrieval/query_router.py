TEMPORAL_TERMS = {
    "today",
    "latest",
    "current",
    "recent",
    "recently",
    "now",
    "this week",
    "this month",
    "breaking",
    "news",
    "update",
}

DOCUMENT_TERMS = {
    "document",
    "pdf",
    "report",
    "notes",
    "file",
    "internal",
    "uploaded",
    "manual",
}



def route_query(query: str) -> str:
    text = query.lower().strip()
    has_temporal = any(term in text for term in TEMPORAL_TERMS)
    has_document = any(term in text for term in DOCUMENT_TERMS)
    asks_compare = any(term in text for term in ["compare", "versus", "vs", "along with"])

    if has_temporal and (has_document or asks_compare):
        return "hybrid"
    if has_temporal:
        return "web"
    if has_document:
        return "document"
    if "http" in text or "wikipedia" in text:
        return "web"
    return "hybrid"
