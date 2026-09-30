# PageIndex on FinanceBench: cost vs accuracy across the latest LLMs

Does [PageIndex](https://github.com/VectifyAI/PageIndex) (vectorless, "reasoning-based" RAG) earn its keep
on financial filings in 2026? This harness answers FinanceBench questions with several retrieval
methods × 36 model/reasoning-effort settings from OpenAI, Anthropic and Google, grades every answer
strictly, and records the exact cost of each one.

## Why

- VectifyAI's headline **98.7% on FinanceBench** ([Mafin 2.5](https://github.com/VectifyAI/Mafin2.5-FinanceBench), Feb 2025)
  counts 12 of 150 answers as correct only after their own human re-grading. Strict agreement with the
  gold answers is 136/150 = **90.7%**. The GPT-4o and DeepSeek-V3 runs have *identical* re-grading labels
  on the same 14 questions despite different answers. Mafin's code is not public.
- Their Aug 2026 [PageIndex Flash](https://pageindex.ai/blog/pageindex-flash) benchmark uses newer models
  (GPT-5.6 Luna/Terra/Sol), but only 62 text-lookup questions that exclude tables and figures (most of a
  10-K), with no vector-RAG baseline and no non-OpenAI models.
- Reddit users report PageIndex as accurate but slow and token-hungry; simpler heading-navigation and
  agentic approaches reach 82–91% on FinanceBench.

## Methods

| Method | What the model sees | One-time cost |
|---|---|---|
| `pageindex` | PageIndex SDK (local mode, Flash tree index); the model searches the tree | Tree per filing, built by `index_model` |
| `full-context` | The whole filing's text in the prompt | None |
| `agentic-search` | Nothing up front; `search` (regex) and `read_pages` tools, like a coding agent | None |
| vector RAG | *Deferred*; see [docs/rag-plan.md](docs/rag-plan.md) | |

All methods share one answer prompt (`finbench/prompts.py`).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an [LLM Foundry](https://llmfoundry.straive.com) key.

```bash
git clone https://github.com/Jivraj-18/benchmark-finance-pageindex && cd benchmark-finance-pageindex
cp .env.example .env            # add LLMFOUNDRY_API_KEY
uv sync
uv run python -m finbench.data download   # 150 questions + 84 PDFs (~160 MB) from patronus-ai/financebench
```

## Run

```bash
# Smoke test: 3 questions, one cheap model
uv run python -m finbench.run --methods pageindex agentic-search --configs gpt-6-luna@low --limit 3

# Pilot: 30 questions, every model/effort setting
uv run python -m finbench.run --methods pageindex agentic-search --configs all --limit 30

# Summarise into results/summary.csv
uv run python -m finbench.report
```

- `--configs` takes `model@effort` settings (`gpt-6-luna@xhigh`, `claude-opus-5.5@low`) or `all` for the grid.
- `--limit N` samples N questions with a fixed seed (`--seed`), so the same N questions every time.
  `--ids` picks specific `financebench_id`s.
- Runs are **resumable**: each (method, config) appends to `results/runs/<method>/<config>.jsonl`, and
  re-running skips questions already answered. Failed calls are recorded with an `error` and retried next run.

Each result line holds the question, gold answer, model answer, token usage, USD cost, latency,
the judge's verdict (`correct` / `incorrect` / `refusal`) and its reason.

## Models

Defined in [`config/models.yaml`](config/models.yaml): the models, their prices and a grid of reasoning
efforts per model (cheap models get the full effort ladder; flagships get the ends).

| Family | Cheap | Mid | Expensive |
|---|---|---|---|
| OpenAI | gpt-6-luna (none → xhigh), gpt-6-luna-pro | gpt-6.1-sol (none → xhigh) | gpt-6-astra (low, high) |
| Anthropic | claude-haiku-4.5 | claude-sonnet-5.5 (minimal → xhigh) | claude-opus-5.5, claude-fable-5.1 |
| Google | gemini-3.1-flash-lite, gemini-3.5-flash-lite | gemini-3.8-flash (minimal → xhigh) | gemini-3.1-pro (low, high) |

**To add a model:** add it under `models:` (OpenRouter id and prices from
`$LLMFOUNDRY_BASE_URL/openrouter/v1/models`), then list its efforts under `grid:`.

## How it works

```
config/models.yaml         models, prices, effort grid, index/judge models
finbench/llm.py            the only module that calls LLMs; meters tokens and cost
finbench/data.py           FinanceBench download, question sampling, per-page PDF text
finbench/prompts.py        shared answer prompt and judge prompt
finbench/methods/*.py      one file per method: answer(question, config) [+ prepare(docs)]
finbench/judge.py          strict grading against the gold answer
finbench/run.py            runs methods × configs × questions in parallel, resumable
finbench/report.py         summary table
```

**LLM access.** Every call goes to LLM Foundry's OpenRouter-compatible route
(`$LLMFOUNDRY_BASE_URL/openrouter/v1`), which serves all three families through one API. PageIndex
calls LiteLLM internally; `llm.pageindex_client()` points it at the same route.

**Cost.** Each call's tokens are priced from `config/models.yaml` (uncached input, cached input, output;
reasoning tokens count as output) and summed per question by `llm.Meter`. PageIndex's one-time tree
building is recorded separately in `results/index_costs.json` and reported per page, not charged to
questions.

**Grading.** One fixed judge (`judge_model`, default `claude-sonnet-5.5@medium`) grades each answer
against the gold answer with the prompt in `finbench/prompts.py`: rounding and formatting differences are
allowed, a different figure or a hedge is `incorrect`, and "cannot find" is `refusal`. No human re-grading.

**Setting.** Each question is answered against its own filing (FinanceBench's "single document"
setting). Finding the right filing among 84 is not tested yet.

**To add a method:** write `finbench/methods/<name>.py` with `answer(question, config) -> {"answer": ...}`
(and optionally `prepare(doc_names)` for one-time indexing), make any LLM calls through `finbench/llm.py`,
and register it in `finbench/methods/__init__.py`.

## Findings from probing LLM Foundry (2026-09-30)

- Claude 5.x and Gemini 3.x reject reasoning effort `none` ("Reasoning is mandatory"); GPT-6.x accepts
  `none` and it really disables thinking.
- OpenAI's managed Files / Vector Stores / `file_search` work on Foundry's `/openai` route, but only with
  gpt-5.4 and older, and Foundry cannot delete uploads. Details in [docs/rag-plan.md](docs/rag-plan.md).
- `pageindex` is pinned to 0.2.20: an unpinned install resolved to a 0.3.0 pre-release with a different API.

## Results

Not run yet at scale. A 3-question smoke test showed PageIndex indexing at **$0.00037 per page** with
gpt-6-luna@low, and agentic search answering for about **$0.0005 per question** with gpt-6-luna, roughly
7× cheaper than PageIndex and 20× cheaper than the whole filing. Three questions are too few to rank anything.
