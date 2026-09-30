"""Baseline: put the whole filing's text in the prompt, then ask the question.

The document comes first and the question last, so a provider's prompt cache
can reuse the document across questions on the same filing.
"""

from ..data import page_texts
from ..llm import complete
from ..prompts import ANSWER_INSTRUCTIONS


def answer(question: dict, config: str) -> dict:
    pages = page_texts(question["doc_name"])
    document = "\n\n".join(f"[Page {i}]\n{text}" for i, text in enumerate(pages, start=1))
    messages = [
        {"role": "system", "content": ANSWER_INSTRUCTIONS},
        {"role": "user", "content": f"<document>\n{document}\n</document>\n\nQuestion: {question['question']}"},
    ]
    return {"answer": complete(config, messages)}
