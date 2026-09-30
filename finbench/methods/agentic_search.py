"""Agentic search: the model explores the filing with grep-style tools, no index.

Like a coding agent exploring a repo: it searches for keywords, reads the pages
that match, and repeats until it can answer. Nothing is built ahead of time.
"""

import re

from ..data import page_texts
from ..llm import run_agent
from ..prompts import ANSWER_INSTRUCTIONS

MAX_MATCHES = 40
MAX_PAGES_PER_READ = 3

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Case-insensitive regex search over the document. Returns matching lines with page numbers.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string", "description": "Python regex, e.g. 'capital expenditure|purchases of property'"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pages",
            "description": f"Full text of pages first..last (at most {MAX_PAGES_PER_READ} pages per call).",
            "parameters": {
                "type": "object",
                "properties": {"first": {"type": "integer"}, "last": {"type": "integer"}},
                "required": ["first", "last"],
            },
        },
    },
]


def answer(question: dict, config: str) -> dict:
    pages = page_texts(question["doc_name"])
    tools = {"search": lambda pattern: search(pages, pattern), "read_pages": lambda first, last: read_pages(pages, first, last)}
    messages = [
        {"role": "system", "content": f"{ANSWER_INSTRUCTIONS}\n\nYou cannot see the document directly. "
                                      f"It has {len(pages)} pages. Use the tools to find the evidence."},
        {"role": "user", "content": question["question"]},
    ]
    text, trace = run_agent(config, messages, tools, SCHEMAS)
    return {"answer": text, "tool_calls": trace}


def search(pages: list[str], pattern: str) -> str:
    regex = re.compile(pattern, re.IGNORECASE)
    matches = [
        f"p{number}: {line.strip()[:200]}"
        for number, text in enumerate(pages, start=1)
        for line in text.splitlines()
        if regex.search(line)
    ]
    if not matches:
        return "No matches."
    more = f"\n... {len(matches) - MAX_MATCHES} more matches; narrow the pattern." if len(matches) > MAX_MATCHES else ""
    return "\n".join(matches[:MAX_MATCHES]) + more


def read_pages(pages: list[str], first: int, last: int) -> str:
    first, last = max(first, 1), min(last, len(pages), first + MAX_PAGES_PER_READ - 1)
    return "\n\n".join(f"[Page {n}]\n{pages[n - 1]}" for n in range(first, last + 1)) or "No such pages."
