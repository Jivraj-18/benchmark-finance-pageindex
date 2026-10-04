"""Every LLM call in this project goes through this module.

Calls go to an LLM proxy chosen by LLM_PROVIDER in .env: `llmfoundry` (default) or `aipipe`.
Both expose the same two routes: chat on an OpenRouter-compatible route, embeddings on an
OpenAI-compatible route. Each call's token usage is priced from config/models.yaml and added to
the active `Meter`, so the cost of any step is whatever was spent inside `with Meter():`.
Inside `with Recorder():`, each chat call's messages and reply are also kept, which is
how the walkthrough page shows what a method did step by step.

Two kinds of callers:
- Our own methods call `complete()`, `run_agent()` (tool calling) and `embed()`.
- PageIndex calls LiteLLM internally. `pageindex_client()` returns a client routed
  through LLM Foundry, and `_wrap_litellm()` wraps LiteLLM so those calls are metered
  and recorded too.
"""

import contextvars
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import APIError, OpenAI, RateLimitError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)  # .env wins over stale shell variables (e.g. an expired token)
CONFIG = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
MODELS = CONFIG["models"]
TIMEOUT = 3600  # seconds per call: give slow, long-thinking calls every chance to finish
# LLM Foundry caches identical requests (response header `x-cache: HIT`, replayed id and text,
# verified 2026-10-04). A benchmark must measure real calls, so every request opts out.
NO_CACHE = {"Cache-Control": "no-cache"}


def resolve(config: str) -> tuple[str, str]:
    """`gpt-6-luna@high` -> ("openai/gpt-6-luna", "high"). No `@` means `default`."""
    model, _, effort = config.partition("@")
    return MODELS[model]["id"], effort or "default"


def grid() -> list[str]:
    """Every `model@effort` config listed in config/models.yaml."""
    return [f"{model}@{effort}" for model, efforts in CONFIG["grid"].items() for effort in efforts]


# LLM_PROVIDER -> (proxy root URL, API key). Each root serves /openrouter/v1 and /openai/v1.
PROVIDERS = {
    "llmfoundry": lambda: (os.environ["LLMFOUNDRY_BASE_URL"], os.environ["LLMFOUNDRY_API_KEY"]),
    "aipipe": lambda: ("https://aipipe.org", os.environ["AIPIPE_TOKEN"]),
}


def _provider() -> tuple[str, str]:
    root, key = PROVIDERS[os.environ.get("LLM_PROVIDER", "llmfoundry")]()
    return root.rstrip("/"), key


def base_url() -> str:
    return _provider()[0] + "/openrouter/v1"


def api_key() -> str:
    return _provider()[1]


def reasoning_body(config: str) -> dict:
    """Request body that sets the reasoning effort, in OpenRouter's `reasoning` form.

    Used for our own calls and PageIndex's, so every call sets effort the same way.
    """
    _, effort = resolve(config)
    return {} if effort == "default" else {"reasoning": {"effort": effort}}


# ---------------------------------------------------------------- metering and recording


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0  # includes cached_tokens
    cached_tokens: int = 0
    output_tokens: int = 0  # includes reasoning tokens
    cost: float = 0.0  # USD
    effort_fallbacks: int = 0  # calls retried at another effort (see _wrap_litellm)

    def to_dict(self) -> dict:
        return asdict(self)


_active_meter: contextvars.ContextVar[Usage | None] = contextvars.ContextVar("meter", default=None)
_active_recording: contextvars.ContextVar[list | None] = contextvars.ContextVar("recording", default=None)


class Meter:
    """Collects the usage of every LLM call made inside the `with` block."""

    def __enter__(self) -> Usage:
        self.usage = Usage()
        self._token = _active_meter.set(self.usage)
        return self.usage

    def __exit__(self, *exc):
        _active_meter.reset(self._token)


class Recorder:
    """Keeps every chat call made inside the `with` block: messages, reply, tokens and seconds.

    Failed calls are kept too, as {"messages", "error", "seconds"}, so a walkthrough shows them.
    """

    def __enter__(self) -> list[dict]:
        self.calls: list[dict] = []
        self._token = _active_recording.set(self.calls)
        return self.calls

    def __exit__(self, *exc):
        _active_recording.reset(self._token)


def price(model_id: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    """USD cost of one call. `model_id` is an OpenRouter id, e.g. openai/gpt-6-luna."""
    p = _prices_by_id()[model_id]
    uncached = input_tokens - cached_tokens
    return (uncached * p["input"] + cached_tokens * p["cache_read"] + output_tokens * p["output"]) / 1e6


@cache
def _prices_by_id() -> dict:
    return {m["id"]: m for m in MODELS.values()}


def _record(model_id: str, usage, messages: list[dict], reply, seconds: float) -> None:
    """Add one chat call to the active Meter and Recorder, if any."""
    meter = _active_meter.get()
    if meter is not None and usage is not None:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        meter.calls += 1
        meter.input_tokens += usage.prompt_tokens
        meter.cached_tokens += cached
        meter.output_tokens += usage.completion_tokens
        meter.cost += price(model_id, usage.prompt_tokens, cached, usage.completion_tokens)
    recording = _active_recording.get()
    if recording is not None:
        out_details = getattr(usage, "completion_tokens_details", None)
        recording.append({
            "model": model_id,
            "messages": list(messages),
            "reply": _as_dict(reply),
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(out_details, "reasoning_tokens", None) if out_details else None,
            "seconds": round(seconds, 2),
        })


def _record_failure(model_id: str, messages: list[dict], error: Exception, seconds: float) -> None:
    """Keep a failed chat call in the active Recorder, if any."""
    recording = _active_recording.get()
    if recording is not None:
        recording.append({"model": model_id, "messages": list(messages), "error": f"{type(error).__name__}: {error}"[:500],
                          "seconds": round(seconds, 2)})


def _as_dict(obj) -> dict:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return dict(obj)


# ---------------------------------------------------------------- our calls


@cache
def _client() -> OpenAI:
    return OpenAI(base_url=base_url(), api_key=api_key(), timeout=TIMEOUT, default_headers=NO_CACHE)


@cache
def _embedding_client() -> OpenAI:
    """The proxy's OpenAI route: embeddings are not on the OpenRouter route."""
    return OpenAI(base_url=_provider()[0] + "/openai/v1", api_key=api_key(), timeout=TIMEOUT,
                  default_headers=NO_CACHE)


def warm_up() -> None:
    """Load the OpenAI SDK's lazily imported modules before worker threads start.

    Threads importing the same module for the first time at once can raise
    `_DeadlockError: deadlock detected by _ModuleLock('openai.resources.chat')`.
    """
    _ = _client().chat.completions, _embedding_client().embeddings


def _retry(call, retries: int = 8, on_failure: Callable[[Exception], None] | None = None):
    """Retry rate limits and server errors with exponential backoff (1, 2, 4 … 60 s)."""
    for attempt in range(retries):
        try:
            return call()
        except (RateLimitError, APIError) as e:
            if on_failure:
                on_failure(e)
            if attempt == retries - 1 or getattr(e, "status_code", 500) in (400, 401, 403, 404):
                raise
            time.sleep(min(2**attempt, 60))


def _chat(config: str, messages: list[dict], **kwargs):
    """One metered chat completion. Returns the assistant message object."""
    model_id, _ = resolve(config)
    if body := reasoning_body(config):
        kwargs["extra_body"] = body
    start = time.time()

    def failed(error: Exception) -> None:
        nonlocal start
        _record_failure(model_id, messages, error, time.time() - start)
        start = time.time()

    response = _retry(lambda: _client().chat.completions.create(model=model_id, messages=messages, **kwargs),
                      on_failure=failed)
    message = response.choices[0].message
    _record(model_id, response.usage, messages, message, time.time() - start)
    return message


def complete(config: str, messages: list[dict], **kwargs) -> str:
    """Chat completion with a `model@effort` config. Returns the text."""
    return _chat(config, messages, **kwargs).content or ""


def run_agent(config: str, messages: list[dict], tools: dict[str, Callable[..., str]], schemas: list[dict],
              max_turns: int = 15) -> tuple[str, list[dict]]:
    """Tool-calling loop. `tools` maps a tool name to the Python function that runs it.

    Returns (final answer, list of tool calls made). If the model is still calling
    tools after `max_turns`, it is asked to answer with what it has.
    """
    messages, trace = list(messages), []
    for _ in range(max_turns):
        message = _chat(config, messages, tools=schemas)
        # model_dump keeps provider extras (e.g. reasoning details) needed on the next turn
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content or "", trace
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            trace.append({"tool": call.function.name, "args": args})
            try:
                result = tools[call.function.name](**args)
            except Exception as e:  # tell the model, let it recover
                result = f"Error: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    messages.append({"role": "user", "content": "Stop searching. Answer now with what you have found."})
    return complete(config, messages), trace


def embed(texts: list[str], batch: int = 256) -> list[list[float]]:
    """Embeddings from config `embedding_model`, metered like chat calls."""
    model = CONFIG["embedding_model"]
    vectors = []
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        response = _retry(lambda: _embedding_client().embeddings.create(model=model["id"], input=chunk))
        vectors += [item.embedding for item in response.data]
        meter = _active_meter.get()
        if meter is not None:
            meter.calls += 1
            meter.input_tokens += response.usage.prompt_tokens
            meter.cost += response.usage.prompt_tokens * model["input"] / 1e6
    return vectors


# ---------------------------------------------------------------- PageIndex


def pageindex_client(storage_path: Path):
    """A local-mode PageIndexClient whose index and chat calls go through LLM Foundry."""
    from pageindex import PageIndexClient

    _wrap_litellm()
    backend = {"api_key": api_key(), "api_base": base_url()}
    index = pageindex_chat_args(CONFIG["index_model"])
    return PageIndexClient(
        index_model=index["model"],
        chat_model=index["model"],  # every chat() call passes its own model
        # index_backend is passed verbatim to every indexing litellm call
        index_backend=backend | {"timeout": TIMEOUT, "extra_body": index["extra_body"], "extra_headers": NO_CACHE},
        chat_backend=backend,
        storage_path=str(storage_path),
        summary_concurrency=16,  # PageIndex default is 64 parallel calls per document; be gentler on LLM Foundry
    )


def pageindex_chat_args(config: str) -> dict:
    """`model`, `extra_body` (reasoning effort) and `extra_headers` arguments for PageIndexClient.chat()."""
    model_id, _ = resolve(config)
    return {"model": "openrouter/" + model_id, "extra_body": reasoning_body(config), "extra_headers": NO_CACHE}


# Through LLM Foundry/OpenRouter, gpt-6-luna answers some prompts with an instant 504
# "The operation was aborted", every time, at one effort level but not at others
# (verified 2026-09-30: a prompt failing at `low` succeeds at `none` and `medium`, and
# vice versa). Such a call is retried once at the neighbouring effort below and counted
# in Usage.effort_fallbacks, rather than failing the whole document or question.
FALLBACK_EFFORT = {"none": "low", "minimal": "low", "low": "medium", "medium": "low", "high": "medium", "xhigh": "high"}


def _fallback_kwargs(kwargs: dict, error: Exception) -> dict | None:
    """The same call at the fallback effort, if `error` is that 504 and an effort was set."""
    effort = ((kwargs.get("extra_body") or {}).get("reasoning") or {}).get("effort")
    if "The operation was aborted" not in str(error) or effort not in FALLBACK_EFFORT:
        return None
    extra_body = kwargs["extra_body"] | {"reasoning": {"effort": FALLBACK_EFFORT[effort]}}
    return kwargs | {"extra_body": extra_body}


@cache
def _wrap_litellm() -> None:
    """Wrap litellm.(a)completion so PageIndex's calls are metered, recorded, and get the effort fallback."""
    import logging

    import litellm

    # LiteLLM's background logging tasks are cancelled at every asyncio.run() exit; asyncio
    # reports each one as "Task was destroyed but it is pending". Harmless, so silence it.
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    completion, acompletion = litellm.completion, litellm.acompletion

    def model_id(kwargs: dict) -> str:
        return kwargs["model"].removeprefix("openrouter/")

    def after(kwargs: dict, response, fell_back: bool, start: float):
        _record(model_id(kwargs), getattr(response, "usage", None), kwargs.get("messages", []),
                response.choices[0].message, time.time() - start)
        meter = _active_meter.get()
        if fell_back and meter is not None:
            meter.effort_fallbacks += 1
        return response

    def failed(kwargs: dict, error: Exception, start: float) -> dict | None:
        """Log the failure; return the fallback call's kwargs, or None to re-raise."""
        _record_failure(model_id(kwargs), kwargs.get("messages", []), error, time.time() - start)
        return _fallback_kwargs(kwargs, error)

    def wrapped_completion(*args, **kwargs):
        start = time.time()
        try:
            return after(kwargs, completion(*args, **kwargs), False, start)
        except Exception as e:
            if (retry := failed(kwargs, e, start)) is None:
                raise
            start = time.time()
            return after(retry, completion(*args, **retry), True, start)

    async def wrapped_acompletion(*args, **kwargs):
        start = time.time()
        try:
            return after(kwargs, await acompletion(*args, **kwargs), False, start)
        except Exception as e:
            if (retry := failed(kwargs, e, start)) is None:
                raise
            start = time.time()
            return after(retry, await acompletion(*args, **retry), True, start)

    litellm.completion, litellm.acompletion = wrapped_completion, wrapped_acompletion
