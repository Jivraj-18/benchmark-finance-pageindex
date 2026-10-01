"""FinanceBench questions, PDFs, and per-page text.

Download everything the benchmark needs (150 questions and the 84 PDFs they cite):
    uv run python -m finbench.data download
"""

import json
import random
from functools import cache
from pathlib import Path

import pypdfium2
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PDFS = DATA / "pdfs"
CACHE = ROOT / "cache"
QUESTIONS_FILE = DATA / "financebench_open_source.jsonl"
SOURCE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS_FILE.read_text().splitlines()]


def select(questions: list[dict], limit: int | None = None, ids: list[str] | None = None, seed: int = 0) -> list[dict]:
    """A reproducible subset: explicit `ids`, or a seeded random sample of `limit` questions."""
    if ids:
        wanted = set(ids)
        return [q for q in questions if q["financebench_id"] in wanted]
    if limit:
        return sorted(random.Random(seed).sample(questions, limit), key=lambda q: q["financebench_id"])
    return questions


def pdf_path(doc_name: str) -> Path:
    return PDFS / f"{doc_name}.pdf"


@cache
def page_texts(doc_name: str) -> list[str]:
    """Text of each page (index 0 = page 1), extracted once and cached as JSON."""
    cached = CACHE / "pages" / f"{doc_name}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    pdf = pypdfium2.PdfDocument(pdf_path(doc_name))
    pages = [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(pages))
    return pages


def download() -> None:
    """Fetch the question file and every PDF the questions refer to."""
    DATA.mkdir(exist_ok=True)
    PDFS.mkdir(exist_ok=True)
    _fetch(f"{SOURCE}/data/{QUESTIONS_FILE.name}", QUESTIONS_FILE)
    docs = sorted({q["doc_name"] for q in load_questions()})
    for doc in tqdm(docs, desc="PDFs"):
        _fetch(f"{SOURCE}/pdfs/{doc}.pdf", pdf_path(doc))
    print(f"{len(load_questions())} questions, {len(docs)} PDFs in {DATA}")


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    dest.write_bytes(response.content)


def question_list(questions: list[dict], title: str) -> str:
    """Markdown table of questions with their filings and gold answers."""
    def cell(text: str) -> str:
        return " ".join(str(text).split()).replace("|", "\\|")

    lines = [
        f"# {title}",
        "",
        f"{len(questions)} questions from [FinanceBench](https://github.com/patronus-ai/financebench) "
        f"(open-source set), over {len({q['doc_name'] for q in questions})} filings.",
        "",
        "| # | ID | Company | Filing | Pages | Type | Question | Gold answer |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n, q in enumerate(questions, start=1):
        pages = len(page_texts(q["doc_name"])) if pdf_path(q["doc_name"]).exists() else ""
        lines.append(f"| {n} | `{q['financebench_id']}` | {cell(q['company'])} | {q['doc_name']} | {pages} | "
                     f"{q['question_type']} | {cell(q['question'])} | {cell(q['answer'])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="python -m finbench.data")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="fetch the questions and PDFs")
    listing = commands.add_parser("list", help="write a Markdown list of the questions a run uses")
    listing.add_argument("--limit", type=int, help="same seeded sample as finbench.run --limit")
    listing.add_argument("--seed", type=int, default=0)
    listing.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "download":
        download()
    else:
        chosen = select(load_questions(), args.limit, seed=args.seed)
        title = f"Questions: --limit {args.limit} --seed {args.seed}" if args.limit else "Questions: all"
        args.out.write_text(question_list(chosen, title))
        print(f"Wrote {len(chosen)} questions to {args.out}")
