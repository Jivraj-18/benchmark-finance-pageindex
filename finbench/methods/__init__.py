"""Answering methods.

Each method module has:
    answer(question: dict, config: str) -> {"answer": str, ...}
and optionally:
    prepare(doc_names: list[str]) -> {doc_name: one-time cost record}
for per-document work (indexing) that is paid once, not per question.

The runner meters LLM cost around each call, so methods don't track cost themselves.
To add a method: write a module here and register it below.
"""

from . import agentic_search, full_context, pageindex_rag, vector_rag

METHODS = {
    "full-context": full_context,
    "pageindex": pageindex_rag,
    "vector-rag": vector_rag,
    "agentic-search": agentic_search,
}
