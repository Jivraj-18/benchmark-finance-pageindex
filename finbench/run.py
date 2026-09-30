"""Run methods x model configs over FinanceBench questions, grade, and save.

Examples:
    uv run python -m finbench.run --methods pageindex --configs gpt-6-luna@low --limit 5
    uv run python -m finbench.run --methods pageindex full-context --configs all --limit 30

One JSONL file per (method, config) in results/runs/. Re-running skips questions
already answered, so an interrupted run resumes where it stopped.
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .data import ROOT, load_questions, select
from .judge import grade
from .llm import Meter, grid
from .methods import METHODS

RUNS = ROOT / "results" / "runs"
INDEX_COSTS = ROOT / "results" / "index_costs.json"
_write_lock = threading.Lock()


def run_file(method: str, config: str) -> Path:
    return RUNS / method / f"{config}.jsonl"


def done_ids(path: Path) -> set[str]:
    """Questions answered without error. Errored ones are retried on the next run."""
    if not path.exists():
        return set()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"] for r in records if "error" not in r}


def run_one(method: str, config: str, question: dict) -> dict:
    record = {
        "id": question["financebench_id"],
        "doc": question["doc_name"],
        "question": question["question"],
        "gold": question["answer"],
        "method": method,
        "config": config,
    }
    start = time.time()
    try:
        with Meter() as usage:
            record |= METHODS[method].answer(question, config)
        record |= {"usage": usage.to_dict(), "seconds": round(time.time() - start, 1)}
        record |= grade(question["question"], question["answer"], record["answer"])
    except Exception as e:  # a failed call is data too: record it and move on
        record |= {"error": f"{type(e).__name__}: {e}"[:1000], "seconds": round(time.time() - start, 1)}
    return record


def prepare(method: str, questions: list[dict]) -> None:
    """One-time per-document work (e.g. PageIndex trees), with its cost saved separately."""
    module = METHODS[method]
    if not hasattr(module, "prepare"):
        return
    docs = sorted({q["doc_name"] for q in questions})
    costs = json.loads(INDEX_COSTS.read_text()) if INDEX_COSTS.exists() else {}
    costs[method] = costs.get(method, {}) | module.prepare(docs)
    INDEX_COSTS.parent.mkdir(parents=True, exist_ok=True)
    INDEX_COSTS.write_text(json.dumps(costs, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--configs", nargs="+", required=True, help="model@effort configs, or 'all' for the grid")
    parser.add_argument("--limit", type=int, help="random sample of N questions (seeded, reproducible)")
    parser.add_argument("--ids", nargs="+", help="specific financebench_ids")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    configs = grid() if args.configs == ["all"] else args.configs
    questions = select(load_questions(), args.limit, args.ids, args.seed)
    print(f"{len(questions)} questions x {len(args.methods)} methods x {len(configs)} configs")

    for method in args.methods:
        prepare(method, questions)

    jobs = []
    for method in args.methods:
        for config in configs:
            path = run_file(method, config)
            finished = done_ids(path)
            jobs += [(method, config, q, path) for q in questions if q["financebench_id"] not in finished]

    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_one, m, c, q): path for m, c, q, path in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="answers"):
            path = futures[future]
            with _write_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as f:
                    f.write(json.dumps(future.result()) + "\n")


if __name__ == "__main__":
    main()
