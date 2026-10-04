"""Record what each method does, step by step, for a few questions.

    uv run python -m finbench.walkthrough --ids financebench_id_00302 --config gpt-6-luna@high
    uv run python -m finbench.walkthrough --limit 30 --config gpt-6-luna@high   # the pilot's questions

Runs every method on each question with one setting, keeps every LLM call (messages,
reply, tokens), grades the answer, and saves one JSON per method to
results/walkthroughs/<question id>/<config>/<method>.json. `finbench.site` turns
these into the walkthrough page. Re-running skips what is already recorded (unless --force).
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .data import ROOT, load_questions, select
from .judge import grade
from .llm import Meter, Recorder, warm_up
from .methods import METHODS
from .run import prepare

WALKTHROUGHS = ROOT / "results" / "walkthroughs"
MAX_CHARS = 4000  # long message contents (whole filings, page text) are cut to this
# Not a result of the method, so not saved (retried on the next run): rate limits and exhausted
# credit at the LLM proxy, and Python import deadlocks between threads
NOT_A_RESULT = re.compile(r"too_many_errors|Too many requests|Error code: 402|requires more credits|RateLimitError|_DeadlockError")


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
        except Exception as e:  # a method's own failure is part of the walkthrough...
            if NOT_A_RESULT.search(f"{type(e).__name__}: {e}"):  # ...but infrastructure failures are not
                raise
            result = {"error": f"{type(e).__name__}: {e}"}
    out = {
        "id": question["financebench_id"],
        "question": question["question"],
        "gold": question["answer"],
        "doc": question["doc_name"],
        "method": method,
        "config": config,
        "provider": os.environ.get("LLM_PROVIDER", "llmfoundry"),
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
    parser.add_argument("--ids", nargs="+", help="specific financebench_ids")
    parser.add_argument("--limit", type=int, help="same seeded sample as finbench.run --limit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", required=True, help="one model@effort setting")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--workers", type=int, default=4, help="questions recorded in parallel")
    parser.add_argument("--force", action="store_true", help="re-record questions that already have a walkthrough")
    args = parser.parse_args()
    if not (args.ids or args.limit):
        parser.error("give --ids or --limit")

    questions = select(load_questions(), args.limit, args.ids, args.seed)
    for method in args.methods:
        prepare(method, questions)
    warm_up()

    def record_question(q: dict) -> None:
        for method in args.methods:  # methods in sequence, so each one's timing is not skewed by the others
            path = WALKTHROUGHS / q["financebench_id"] / args.config / f"{method}.json"
            if path.exists() and not args.force:
                continue
            try:
                out = record(q, method, args.config)
            except Exception as e:  # e.g. grading rate-limited: skip, and the next run records it
                print(f"{q['financebench_id']} {method:<15} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=1))
            print(f"{q['financebench_id']} {method:<15} {out.get('verdict', 'error'):<10} "
                  f"${out['usage']['cost']:.4f} {out['usage']['calls']} calls {out['seconds']}s", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(record_question, questions))


if __name__ == "__main__":
    main()
