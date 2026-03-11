from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationScenario:
    category: str
    prompt: str
    expected_route: str
    goal: str


TEST_SCENARIOS = [
    EvaluationScenario(
        category="document",
        prompt="What do the uploaded transformer notes say about attention?",
        expected_route="document",
        goal="Check grounded retrieval over indexed technical notes.",
    ),
    EvaluationScenario(
        category="document",
        prompt="What is the role of the Supreme Court in protecting fundamental rights?",
        expected_route="document",
        goal="Check retrieval over legal PDF content with chunk citations.",
    ),
    EvaluationScenario(
        category="web",
        prompt="What are the latest developments in retrieval augmented generation systems?",
        expected_route="web",
        goal="Check current-information routing and Tavily evidence separation.",
    ),
    EvaluationScenario(
        category="web",
        prompt="What is the latest news about Groq and open-weight LLM serving?",
        expected_route="web",
        goal="Check live web retrieval quality for provider-specific updates.",
    ),
    EvaluationScenario(
        category="hybrid",
        prompt="How do the uploaded transformer notes compare with current RAG tooling trends?",
        expected_route="hybrid",
        goal="Check mixed internal and external evidence assembly.",
    ),
    EvaluationScenario(
        category="hybrid",
        prompt="Compare the indexed court material with current legal commentary on fundamental rights.",
        expected_route="hybrid",
        goal="Check balanced hybrid reasoning with clear source separation.",
    ),
]


def scenarios_by_category() -> dict[str, list[EvaluationScenario]]:
    buckets: dict[str, list[EvaluationScenario]] = {"document": [], "web": [], "hybrid": []}
    for item in TEST_SCENARIOS:
        buckets.setdefault(item.category, []).append(item)
    return buckets
