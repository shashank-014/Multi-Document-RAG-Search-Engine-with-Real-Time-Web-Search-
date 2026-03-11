import re

_ARTIFACT_PATTERNS = [
    re.compile(r"\f"),
    re.compile(r"[ \t]+"),
    re.compile(r"\n{3,}"),
    re.compile(r"(?m)^\s*[\-_]{3,}\s*$"),
]



def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()



def cleanup_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text)



def filter_artifacts(text: str) -> str:
    cleaned = text
    for pattern in _ARTIFACT_PATTERNS:
        if pattern.pattern == r"[ \t]+":
            continue
        if pattern.pattern == r"\n{3,}":
            continue
        cleaned = pattern.sub("\n", cleaned)
    cleaned = re.sub(r"(?m)^\s*Page\s+\d+\s*$", "", cleaned)
    return cleaned



def clean_text(text: str) -> str:
    text = filter_artifacts(text)
    text = cleanup_newlines(text)
    return normalize_whitespace(text.replace("\n ", "\n"))
