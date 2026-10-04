# Vector RAG baseline

Status (2026-10-04): **Option B is implemented** as the `vector-rag` method (`finbench/methods/vector_rag.py`).
Option A (OpenAI's managed vector store) is not; the notes below stay for when it is wanted.
Everything below was verified by probing LLM Foundry on 2026-09-30.

## Why it matters

Vector RAG is what PageIndex claims to beat ("similarity ≠ relevance"). Without it, the benchmark
can only compare PageIndex to non-vector baselines (whole filing in context, agentic search).

## Option A: OpenAI vector store (managed chunking + embedding)

Upload PDFs; OpenAI chunks, embeds and indexes them.

What works through LLM Foundry (`$LLMFOUNDRY_BASE_URL/openai/v1`):
- `POST /files` (purpose `assistants`) and `POST /vector_stores`: upload and index. Indexing is free;
  storage is $0.10/GB/day beyond 1 GB free.
- `POST /vector_stores/{id}/search`: returns chunks + scores, so **any model** (Claude, Gemini, GPT-6)
  can answer from them. Chunks carry no page numbers (`attributes` is empty) unless we add them.
- Responses API with the `file_search` tool (fully managed, one call): works only with
  **gpt-5.4, gpt-5.4-mini, gpt-4.1-mini**.

What doesn't work:
- GPT-5.5 / 5.6 / 6.x on the `/openai` route: `Model ... is not enabled on LLM Foundry. Contact admin to add it.`
  These models are only on Foundry's `/openrouter` route, and OpenRouter has no Files / Vector Stores API.
  **Ask the Foundry admin to enable gpt-6.1-sol and gpt-6-luna on `/openai`** to run managed file_search on them.
- `DELETE` on files and vector stores: Foundry's proxy returns a Python traceback, so uploads cannot be removed.
  Create stores with `expires_after={"anchor": "last_active_at", "days": 7}` so storage never bills.
  A 13 KB probe file (`FOOTLOCKER_2022_8K`) and its store from 2026-09-30 are still on the account.

Costs to record: $2.50 per 1,000 file_search calls. The pricing page does not list the standalone
search endpoint; assume the same rate until confirmed. gpt-5.4: $2.50 / $15 per M tokens;
gpt-5.4-mini: $0.75 / $4.50 (input / output).

## Option B: our own vector store

Model-agnostic and fully under our control (page numbers, chunk size, hybrid BM25, reranking).
- Embeddings work through Foundry: `POST $LLMFOUNDRY_BASE_URL/openai/v1/embeddings`,
  `text-embedding-3-large` (3,072 dims, $0.13 per M tokens). `text-embedding-3-small` is also listed.
- A median filing is ~110k tokens, so embedding all 84 filings costs about $1.
- Needs `numpy` (cosine similarity) and optionally `rank-bm25` for a hybrid variant.

## How to add either one to the harness

1. Put every API call (upload, search, embed) in `finbench/llm.py` and meter it: add token cost
   and any per-search fee to the active `Meter`.
2. Write `finbench/methods/<name>.py` with `prepare(doc_names)` (one-time indexing, cost recorded in
   `results/index_costs.json`) and `answer(question, config)`.
3. Register it in `finbench/methods/__init__.py`. `finbench.run` then runs it over the model grid.
