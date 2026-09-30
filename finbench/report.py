"""Summarise results/runs/ into one row per (method, config).

    uv run python -m finbench.report

Writes results/summary.csv and prints it. When a question was answered more than
once (e.g. after a retry), the last record counts.
"""

import csv
import json
from statistics import mean, median

from .data import ROOT
from .llm import MODELS
from .run import INDEX_COSTS, RUNS

SUMMARY = ROOT / "results" / "summary.csv"


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
        graded = [r for r in group if "verdict" in r]
        verdicts = [r["verdict"] for r in graded]
        model = config.partition("@")[0]
        rows.append({
            "method": method,
            "config": config,
            "family": MODELS[model]["family"],
            "questions": len(graded),
            "errors": len(group) - len(graded),
            "accuracy_pct": round(100 * verdicts.count("correct") / len(graded), 1) if graded else None,
            "refusal_pct": round(100 * verdicts.count("refusal") / len(graded), 1) if graded else None,
            "cost_per_q_usd": round(mean(r["usage"]["cost"] for r in graded), 5) if graded else None,
            "median_seconds": median(r["seconds"] for r in graded) if graded else None,
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


def main() -> None:
    rows = summarise(load_runs())
    if not rows:
        raise SystemExit("No results yet. Run finbench.run first.")
    with SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'method':<16}{'config':<30}{'n':>4}{'acc%':>7}{'refuse%':>9}{'$/question':>12}{'med s':>7}")
    for r in rows:
        print(f"{r['method']:<16}{r['config']:<30}{r['questions']:>4}{r['accuracy_pct']:>7}"
              f"{r['refusal_pct']:>9}{r['cost_per_q_usd']:>12.5f}{r['median_seconds']:>7}")
    for method, cost in index_cost_per_page().items():
        print(f"\n{method} indexing: ${cost:.5f} per page (one-time)")
    print(f"\nWrote {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
