"""Strict LLM grading of answers against FinanceBench gold answers."""

import json
import re

from .llm import CONFIG, Meter, complete
from .prompts import JUDGE

VERDICTS = ("correct", "incorrect", "refusal")


def grade(question: str, gold: str, answer: str, config: str = CONFIG["judge_model"]) -> dict:
    """{"verdict", "reason", "judge_cost"} for one answer."""
    prompt = JUDGE.format(question=question, gold=gold, answer=answer)
    with Meter() as usage:
        reply = complete(config, [{"role": "user", "content": prompt}])
    verdict = _parse(reply)
    return {**verdict, "judge_model": config, "judge_cost": usage.cost}


def _parse(reply: str) -> dict:
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    verdict = data.get("verdict")
    if verdict not in VERDICTS:
        return {"verdict": "unparsed", "reason": reply[:500]}
    return {"verdict": verdict, "reason": data.get("reason", "")}
