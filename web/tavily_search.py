import os

from langchain_community.tools.tavily_search import TavilySearchResults

from config import DEFAULT_TOP_K, get_secret


def search_web(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, str]]:
    api_key = get_secret("TAVILY_API_KEY")
    if not api_key:
        return []

    os.environ["TAVILY_API_KEY"] = api_key
    tool = TavilySearchResults(max_results=top_k)
    raw_results = tool.invoke({"query": query})

    structured = []
    for index, item in enumerate(raw_results or [], start=1):
        structured.append(
            {
                "source_id": item.get("url") or f"tavily-result-{index}",
                "title": item.get("title", "Untitled web result"),
                "snippet": item.get("content", ""),
                "url": item.get("url", ""),
                "source_type": "web",
            }
        )

    return structured
