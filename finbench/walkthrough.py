"""Record what each method does, step by step, for a few questions.

    uv run python -m finbench.walkthrough --ids financebench_id_00302 --config gpt-6-luna@high

Runs every method on each question with one setting, keeps every LLM call (messages,
reply, tokens), grades the answer, and saves one JSON per method to
results/walkthroughs/<question id>/<config>/<method>.json. `finbench.site` turns
these into the walkthrough page.
"""

import argparse
import json
import time

from .data import ROOT, load_questions
from .judge import grade
from .llm import Meter, Recorder
from .methods import METHODS
from .run import prepare

WALKTHROUGHS = ROOT / "results" / "walkthroughs"
MAX_CHARS = 4000  # long message contents (whole filings, page text) are cut to this


def shorten(value):
    """Cut long strings anywhere inside a message, noting how much was dropped."""
    if isinstance(value, str) and len(value) > MAX_CHARS:
        return value[:MAX_CHARS] + f"\n… [{len(value) - MAX_CHARS:,} more characters]"
    if isinstance(value, list):
        return [shorten(v) for v in value]
    if isinstance(value, dict):
        return {k: shorten(v) for k, v in value.items()}
    return value


def record(question: dict, method: str, config: str) -> dict:
    start = time.time()
    with Meter() as usage, Recorder() as calls:
        try:
            result = METHODS[method].answer(question, config)
        except Exception as e:  # a failure is part of the walkthrough
            result = {"error": f"{type(e).__name__}: {e}"}
    out = {
        "id": question["financebench_id"],
        "question": question["question"],
        "gold": question["answer"],
        "doc": question["doc_name"],
        "method": method,
        "config": config,
        "seconds": round(time.time() - start, 1),
        "usage": usage.to_dict(),
        "calls": shorten(calls),
        **shorten(result),
    }
    if "answer" in result:
        out |= grade(question["question"], question["answer"], result["answer"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--config", required=True, help="one model@effort setting")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    args = parser.parse_args()

    questions = [q for q in load_questions() if q["financebench_id"] in set(args.ids)]
    for method in args.methods:
        prepare(method, questions)
    for q in questions:
        for method in args.methods:
            out = record(q, method, args.config)
            path = WALKTHROUGHS / q["financebench_id"] / args.config / f"{method}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=1))
            print(f"{q['financebench_id']} {method:<15} {out.get('verdict', 'error'):<10} "
                  f"${out['usage']['cost']:.4f} {out['usage']['calls']} calls {out['seconds']}s")


if __name__ == "__main__":
    main()
