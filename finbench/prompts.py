"""Prompts shared across methods, so every method is asked the same way."""

ANSWER_INSTRUCTIONS = """\
You are a financial analyst answering a question about one SEC filing or earnings document.
Use only the document. Give the final answer first, then a short justification citing page numbers.
Show the calculation when the question asks for a ratio, margin, or change.
If the document does not contain what is needed, reply exactly: I cannot find this in the document."""

JUDGE = """\
You are grading an answer to a financial question against an expert's gold answer.

Question: {question}
Gold answer: {gold}
Answer to grade: {answer}

Rules:
- "correct": the answer reaches the same conclusion or figure as the gold answer.
  Rounding differences, unit formatting ($1,577M vs $1.577B) and extra supporting detail are fine.
- "incorrect": a different figure or conclusion, a missing key part of the gold answer,
  or a hedge that does not commit to the gold answer's conclusion.
- "refusal": the answer says it cannot find or cannot answer.
Grade strictly against the gold answer, even if you think the gold answer is debatable.

Reply with JSON only: {{"verdict": "correct" | "incorrect" | "refusal", "reason": "<one sentence>"}}"""
