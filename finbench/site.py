"""Build the GitHub Pages site in docs/ from results.

    uv run python -m finbench.site --limit 30 --walkthrough-config gpt-6-luna@high \
        --walkthrough-order financebench_id_04103 financebench_id_00302 financebench_id_01107 financebench_id_00521

docs/index.html        cost vs accuracy: one dot per method x setting
docs/walkthrough.html  each method's steps for the questions in results/walkthroughs/

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


def walkthroughs(config: str, order: list[str] | None = None) -> list[dict]:
    """[{question, gold, doc, methods: [{method, verdict, cost, calls, seconds, steps}]}] for one setting.

    Questions follow `order` (question ids) where given, then the rest by id.
    """
    out = []
    folders = sorted(WALKTHROUGHS.glob(f"*/{config}"), key=lambda f: (f.parent.name not in (order or []),
                     (order or []).index(f.parent.name) if f.parent.name in (order or []) else 0, f.parent.name))
    for folder in folders:
        traces = {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")}
        if not traces:
            continue
        first = next(iter(traces.values()))
        out.append({
            "id": first["id"], "question": first["question"], "gold": first["gold"], "doc": first["doc"],
            "methods": [
                {"method": m, "verdict": t.get("verdict", "error"), "reason": t.get("reason", ""),
                 "cost": t["usage"]["cost"], "calls": t["usage"]["calls"],
                 "input_tokens": t["usage"]["input_tokens"], "seconds": t["seconds"], "steps": steps(t)}
                for m in METHODS if (t := traces.get(m))
            ],
        })
    return out


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
    parser.add_argument("--walkthrough-order", nargs="+", help="question ids, in tab order (first one opens by default)")
    args = parser.parse_args()

    meta = {"questions": args.limit or len(load_questions()), "settings": grid(), "methods": list(METHODS),
            "judge": CONFIG["judge_model"], "index_model": CONFIG["index_model"],
            "repo": "https://github.com/Jivraj-18/benchmark-finance-pageindex"}
    SITE.mkdir(exist_ok=True)
    (SITE / "style.css").write_text((TEMPLATES / "style.css").read_text())
    (SITE / "index.html").write_text(render("index.html", {"meta": meta, "cells": chart_cells(args.limit, args.seed)}))
    walks = walkthroughs(args.walkthrough_config, args.walkthrough_order)
    (SITE / "walkthrough.html").write_text(
        render("walkthrough.html", {"meta": meta | {"config": args.walkthrough_config}, "questions": walks}))
    print(f"Wrote {SITE / 'index.html'} and {SITE / 'walkthrough.html'} ({len(walks)} walkthrough questions)")


if __name__ == "__main__":
    main()
