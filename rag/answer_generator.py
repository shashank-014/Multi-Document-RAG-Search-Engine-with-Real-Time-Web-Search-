import os

import streamlit as st

from groq import BadRequestError, Groq

api_key = st.secrets.get("GROQ_API_KEY")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key

MAX_CONTEXT_CHARS = 2600
MAX_MEMORY_CHARS = 800
GROQ_MODEL = "llama-3.1-8b-instant"


def _trim_text(text, limit):
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _build_messages(query, context, memory_text):
    safe_context = _trim_text(context, MAX_CONTEXT_CHARS)
    safe_memory = _trim_text(memory_text, MAX_MEMORY_CHARS)

    system_prompt = f"""
You are a helpful AI assistant.

Use only the provided context to answer the user's question.
Keep the answer grounded and concise.
If the answer is not present in the context, say you do not know.
When evidence is available, keep the source tags exactly as written in the context.

Context:
{safe_context}

Conversation History:
{safe_memory}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]


def _run_completion(messages):
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _attach_citations(answer_text, citations):
    if not citations:
        return answer_text

    trimmed = (answer_text or "").strip()
    citation_block = "Sources: " + "; ".join(citations[:4])

    if "Sources:" in trimmed:
        return trimmed

    if any(citation in trimmed for citation in citations):
        return trimmed

    if not trimmed:
        return citation_block

    return f"{trimmed}\n\n{citation_block}"


def stream_answer(query, context, memory_text, citations=None):
    if not api_key:
        yield "GROQ_API_KEY is missing in Streamlit secrets."
        return

    if not context.strip():
        fallback = "I could not find enough evidence to answer that question yet."
        if citations:
            fallback = _attach_citations(fallback, citations)
        yield fallback
        return

    messages = _build_messages(query, context, memory_text)

    try:
        answer_text = _run_completion(messages)
    except BadRequestError:
        fallback_messages = _build_messages(
            query,
            _trim_text(context, 1400),
            _trim_text(memory_text, 250),
        )
        try:
            answer_text = _run_completion(fallback_messages)
        except BadRequestError:
            yield "The model request was too large for the current evidence set. Try a shorter question or disable web search."
            return

    if not answer_text:
        answer_text = "I do not know based on the available context."

    answer_text = _attach_citations(answer_text, citations or [])

    for word in answer_text.split():
        yield word + " "


def generate_answer(query, context, memory_text, citations=None):
    return "".join(stream_answer(query, context, memory_text, citations=citations))
