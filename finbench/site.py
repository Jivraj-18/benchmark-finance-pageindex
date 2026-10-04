"""Build the GitHub Pages site in docs/ from results.

    uv run python -m finbench.site --limit 30 --walkthrough-config gpt-6-luna@high \
        --walkthrough-start financebench_id_04103

docs/index.html        cost vs accuracy: one dot per method x setting
docs/walkthrough.html  each method's steps, per question (docs/walkthroughs/<id>.json, loaded on demand)

Templates live in finbench/templates/; this module only prepares their data.
"""

import argparse
import json
from pathlib import Path

from .data import ROOT, load_questions, select
from .llm import CONFIG, grid
from .methods import METHODS
from .report import load_runs, summarise
from .walkthrough import WALKTHROUGHS

TEMPLATES = Path(__file__).parent / "templates"
SITE = ROOT / "docs"
PREVIEW_CHARS = 1200  # tool results and prompts shown on the page are cut to this


# ---------------------------------------------------------------- chart data


def chart_cells(limit: int | None, seed: int) -> list[dict]:
    """One cell per (method, setting) in the grid, over the sampled questions only."""
    ids = {q["financebench_id"] for q in select(load_questions(), limit, seed=seed)}
    records = [r for r in load_runs() if r["id"] in ids and r["config"] in grid() and r["method"] in METHODS]
    return summarise(records)


# ---------------------------------------------------------------- walkthrough steps


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit] + f"\n… [{len(text) - limit:,} more characters]"


def text_of(content) -> str:
    """Message content as plain text (PageIndex tool results are lists of text parts)."""
    if isinstance(content, list):
        return "\n".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content or ""


def thinking_of(reply: dict) -> str:
    """The model's readable thinking summary, if the provider returned one (encrypted parts are skipped)."""
    parts = [d.get("summary") or d.get("text") or "" for d in reply.get("reasoning_details", []) if isinstance(d, dict)]
    text = "\n\n".join(p.strip() for p in parts if p and p.strip())
    return text or (reply.get("reasoning") or reply.get("reasoning_content") or "").strip()


def steps(trace: dict) -> list[dict]:
    """What the method did, in order: each model turn's thinking, tool calls with results, failures, the answer."""
    out = [{"kind": "question", "title": "Gets the question (shown above)"}]
    if trace.get("retrieved"):
        pages = ", ".join(f"p{h['page']} ({h['score']:.2f})" for h in trace["retrieved"])
        out.append({"kind": "retrieve", "title": f"Embeds the question, retrieves the {len(trace['retrieved'])} closest chunks",
                    "body": f"Pages (similarity): {pages}"})
    calls = trace.get("calls", [])
    ok = [c for c in calls if "reply" in c]
    if ok:
        final = ok[-1]
        conversation = final["messages"] + [final["reply"]]
        by_length = {len(c["messages"]): c for c in ok}  # the call that produced the message at index i
        failures: dict[int, list] = {}
        for c in calls:
            if "error" in c:
                failures.setdefault(len(c["messages"]), []).append(c)
        context = [m for m in final["messages"] if m.get("role") in ("system", "user")]
        sent = sum(len(text_of(m.get("content"))) for m in context)
        out.append({"kind": "prompt", "title": f"Sends the model its instructions and context ({sent:,} characters)",
                    "detail": "\n\n".join(f"[{m['role']}]\n{preview(text_of(m.get('content')))}" for m in context)})
        pending = {}
        for i, message in enumerate(conversation):
            role = message.get("role")
            if role == "assistant":
                for f in failures.get(i, []):
                    out.append({"kind": "error", "title": f"Model call fails after {f['seconds']}s, retried",
                                "body": f["error"]})
            call = by_length.get(i) if role == "assistant" else None
            turn = {"thinking": thinking_of(call["reply"]) if call else "",
                    "thought": call.get("reasoning_tokens") if call else None,
                    "tokens": call.get("input_tokens") if call else None,
                    "seconds": call.get("seconds") if call else None}
            if role == "assistant" and message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    step = {"kind": "tool", "title": f"Calls {tc['function']['name']}", "body": tc["function"].get("arguments", ""),
                            **turn}
                    pending[tc.get("id")] = step
                    out.append(step)
                    turn = {}  # a turn's thinking, tokens and time belong to its first tool call
            elif role == "tool":
                step = pending.get(message.get("tool_call_id"))
                if step is not None:
                    step["detail"] = preview(text_of(message.get("content")))
            elif role == "assistant" and i == len(conversation) - 1:
                out.append({"kind": "answer", "title": "Answers", "body": text_of(message.get("content")), **turn})
    if trace.get("error"):
        out.append({"kind": "error", "title": "Fails", "body": trace["error"]})
    return out


def walkthroughs(config: str, questions: list[dict]) -> list[dict]:
    """One entry per recorded question, numbered in sample order, with each method's steps."""
    out = []
    for n, q in enumerate(questions, start=1):
        folder = WALKTHROUGHS / q["financebench_id"] / config
        traces = {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")} if folder.exists() else {}
        if not traces:
            continue
        out.append({
            "n": n, "id": q["financebench_id"], "company": q["company"], "question": q["question"],
            "gold": q["answer"], "doc": q["doc_name"],
            "methods": [
                {"method": m, "verdict": t.get("verdict", "error"), "reason": t.get("reason", ""),
                 "cost": t["usage"]["cost"], "calls": t["usage"]["calls"],
                 "input_tokens": t["usage"]["input_tokens"], "seconds": t["seconds"],
                 "provider": t.get("provider", "llmfoundry"), "steps": steps(t)}
                for m in METHODS if (t := traces.get(m))
            ],
        })
    return out


def question_list(walks: list[dict]) -> list[dict]:
    """The walkthroughs without their steps: what the home table and the question picker need."""
    return [{k: w[k] for k in ("n", "id", "company", "question", "gold")}
            | {"results": {m["method"]: {"verdict": m["verdict"], "cost": m["cost"]} for m in w["methods"]}}
            for w in walks]


# ---------------------------------------------------------------- render


def render(template: str, data: dict) -> str:
    html = (TEMPLATES / template).read_text()
    payload = json.dumps(data).replace("</", "<\\/")  # safe inside <script>
    return html.replace("/*DATA*/null", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="same seeded sample as finbench.run --limit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--walkthrough-config", default="gpt-6-luna@high")
    parser.add_argument("--walkthrough-start", help="question id the walkthrough opens on (default: the first)")
    args = parser.parse_args()

    questions = select(load_questions(), args.limit, seed=args.seed)
    meta = {"questions": len(questions), "settings": grid(), "methods": list(METHODS),
            "judge": CONFIG["judge_model"], "index_model": CONFIG["index_model"],
            "config": args.walkthrough_config, "start": args.walkthrough_start,
            "repo": "https://github.com/Jivraj-18/benchmark-finance-pageindex"}
    walks = walkthroughs(args.walkthrough_config, questions)
    listing = question_list(walks)

    SITE.mkdir(exist_ok=True)
    (SITE / "style.css").write_text((TEMPLATES / "style.css").read_text())
    steps_dir = SITE / "walkthroughs"  # one file per question, loaded when it is opened
    steps_dir.mkdir(exist_ok=True)
    for old in steps_dir.glob("*.json"):
        old.unlink()
    for w in walks:
        (steps_dir / f"{w['id']}.json").write_text(json.dumps(w))
    (SITE / "index.html").write_text(render("index.html", {
        "meta": meta, "cells": chart_cells(args.limit, args.seed), "questions": listing}))
    (SITE / "walkthrough.html").write_text(render("walkthrough.html", {"meta": meta, "questions": listing}))
    print(f"Wrote {SITE / 'index.html'}, {SITE / 'walkthrough.html'} and {len(walks)} walkthroughs in {steps_dir}")


if __name__ == "__main__":
    main()
