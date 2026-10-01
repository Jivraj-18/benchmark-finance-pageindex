"""Summarise results/runs/ into one row per (method, config).

    uv run python -m finbench.report
    uv run python -m finbench.report --answers docs/questions-pilot.md --limit 30

Writes results/summary.csv and prints it. With --answers, also writes a Markdown file
listing each question, its gold answer, and every method/setting's answer and grade.
When a question was answered more than once (e.g. after a retry), the last record counts.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

from .data import ROOT, load_questions, page_texts, select
from .llm import MODELS, grid
from .methods import METHODS
from .run import INDEX_COSTS, RUNS

SUMMARY = ROOT / "results" / "summary.csv"
ANSWER_CHARS = 300  # answers are cut to this length in the answers file; full text is in results/runs/


def load_runs() -> list[dict]:
    records = {}
    for path in sorted(RUNS.glob("*/*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                records[(r["method"], r["config"], r["id"])] = r
    return list(records.values())


def summarise(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["method"], r["config"]), []).append(r)
    rows = []
    for (method, config), group in sorted(groups.items()):
        # Errors that survive a re-run (e.g. filing larger than the context window) count
        # against accuracy: the method failed to answer.
        graded = [r for r in group if "verdict" in r]
        verdicts = [r["verdict"] for r in graded]
        model = config.partition("@")[0]
        rows.append({
            "method": method,
            "config": config,
            "family": MODELS[model]["family"],
            "questions": len(group),
            "errors": len(group) - len(graded),
            "accuracy_pct": round(100 * verdicts.count("correct") / len(group), 1),
            "refusal_pct": round(100 * verdicts.count("refusal") / len(group), 1),
            "cost_per_q_usd": round(mean(r["usage"]["cost"] for r in graded), 5) if graded else None,
            "median_seconds": round(median(r["seconds"] for r in graded), 1) if graded else None,
        })
    return rows


def index_cost_per_page() -> dict:
    """One-time indexing cost per method, per page indexed."""
    from .data import page_texts

    if not INDEX_COSTS.exists():
        return {}
    costs = json.loads(INDEX_COSTS.read_text())
    return {
        method: sum(d["cost"] for d in docs.values()) / sum(len(page_texts(doc)) for doc in docs)
        for method, docs in costs.items()
    }


def answers_markdown(questions: list[dict], records: list[dict], title: str) -> str:
    """Each question with its gold answer, then every method/setting's answer and grade."""
    by_key = {(r["id"], r["method"], r["config"]): r for r in records}
    configs = grid()

    def cell(text: str, limit: int | None = None) -> str:
        text = " ".join(str(text).split()).replace("|", "\\|")
        return text[:limit] + "…" if limit and len(text) > limit else text

    def answers_for(q: dict) -> list[dict]:
        return [by_key[k] for m in METHODS for c in configs if (k := (q["financebench_id"], m, c)) in by_key]

    lines = [
        f"# {title}",
        "",
        f"{len(questions)} questions from [FinanceBench](https://github.com/patronus-ai/financebench) "
        f"(open-source set), over {len({q['doc_name'] for q in questions})} filings.",
        "",
        "The **gold answer** is FinanceBench's reference answer, written by financial analysts from the filing. "
        "Every answer below is graded against it by the judge model in `config/models.yaml`: "
        "**correct**, **incorrect**, **refusal** (said it could not find it) or **error** (the method failed to answer).",
        "",
        f"Methods: {', '.join(METHODS)}. Settings: {', '.join(configs)}. "
        f"Answers are cut to {ANSWER_CHARS} characters; full text is in `results/runs/`.",
        "",
        "## Summary",
        "",
        "| # | Company | Question | Gold answer | Correct |",
        "|---|---|---|---|---|",
    ]
    for n, q in enumerate(questions, start=1):
        rs = answers_for(q)
        correct = sum(r.get("verdict") == "correct" for r in rs)
        lines.append(f"| [{n}](#q{n}) | {cell(q['company'])} | {cell(q['question'])} | {cell(q['answer'])} | {correct}/{len(rs)} |")

    for n, q in enumerate(questions, start=1):
        lines += [
            "",
            f'<a id="q{n}"></a>',
            f"## {n}. {cell(q['company'])}: {cell(q['question'])}",
            "",
            f"`{q['financebench_id']}` · {q['doc_name']} ({len(page_texts(q['doc_name']))} pages) · {q['question_type']}",
            "",
            f"**Gold answer:** {cell(q['answer'])}",
            "",
            "| Method | Setting | Grade | Answer | Cost |",
            "|---|---|---|---|---|",
        ]
        for r in answers_for(q):
            grade = r.get("verdict", "error")
            answer = r.get("answer") or r.get("error", "")
            cost = f"${r['usage']['cost']:.4f}" if "usage" in r else ""
            lines.append(f"| {r['method']} | {r['config']} | **{grade}** | {cell(answer, ANSWER_CHARS)} | {cost} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=Path, help="also write the per-question answers file here")
    parser.add_argument("--limit", type=int, help="questions in the answers file: same seeded sample as finbench.run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = load_runs()
    if args.answers:
        questions = select(load_questions(), args.limit, seed=args.seed)
        title = f"Questions and answers: --limit {args.limit} --seed {args.seed}" if args.limit else "Questions and answers"
        args.answers.write_text(answers_markdown(questions, records, title))
        print(f"Wrote {len(questions)} questions with answers to {args.answers}")

    rows = summarise(records)
    if not rows:
        raise SystemExit("No results yet. Run finbench.run first.")
    with SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'method':<16}{'config':<30}{'n':>4}{'err':>4}{'acc%':>7}{'refuse%':>9}{'$/question':>12}{'med s':>7}")
    for r in rows:
        print(f"{r['method']:<16}{r['config']:<30}{r['questions']:>4}{r['errors']:>4}{r['accuracy_pct']:>7}"
              f"{r['refusal_pct']:>9}{r['cost_per_q_usd'] or 0:>12.5f}{r['median_seconds'] or 0:>7}")
    for method, cost in index_cost_per_page().items():
        print(f"\n{method} indexing: ${cost:.5f} per page (one-time)")
    print(f"\nWrote {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
