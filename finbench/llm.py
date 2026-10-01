"""Every LLM call in this project goes through this module.

Calls are routed to LLM Foundry's OpenRouter-compatible endpoint. Each call's
token usage is priced from config/models.yaml and added to the active `Meter`,
so the cost of any step is whatever was spent inside `with Meter() as m:`.

Two kinds of callers:
- Our own methods call `complete()` and `run_agent()` (tool calling).
- PageIndex calls LiteLLM internally. `pageindex_client()` returns a client
  routed through LLM Foundry, and `_meter_litellm()` wraps LiteLLM so those
  calls are metered too.
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
load_dotenv(ROOT / ".env")
CONFIG = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
MODELS = CONFIG["models"]


def resolve(config: str) -> tuple[str, str]:
    """`gpt-6-luna@high` -> ("openai/gpt-6-luna", "high"). No `@` means `default`."""
    model, _, effort = config.partition("@")
    return MODELS[model]["id"], effort or "default"


def grid() -> list[str]:
    """Every `model@effort` config listed in config/models.yaml."""
    return [f"{model}@{effort}" for model, efforts in CONFIG["grid"].items() for effort in efforts]


def base_url() -> str:
    return os.environ["LLMFOUNDRY_BASE_URL"].rstrip("/") + "/openrouter/v1"


def api_key() -> str:
    return os.environ["LLMFOUNDRY_API_KEY"]


# ---------------------------------------------------------------- metering


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0  # includes cached_tokens
    cached_tokens: int = 0
    output_tokens: int = 0  # includes reasoning tokens
    cost: float = 0.0  # USD

    def to_dict(self) -> dict:
        return asdict(self)


_active_meter: contextvars.ContextVar[Usage | None] = contextvars.ContextVar("meter", default=None)


class Meter:
    """Collects the usage of every LLM call made inside the `with` block."""

    def __enter__(self) -> Usage:
        self.usage = Usage()
        self._token = _active_meter.set(self.usage)
        return self.usage

    def __exit__(self, *exc):
        _active_meter.reset(self._token)


def price(model_id: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    """USD cost of one call. `model_id` is an OpenRouter id, e.g. openai/gpt-6-luna."""
    p = _prices_by_id()[model_id]
    uncached = input_tokens - cached_tokens
    return (uncached * p["input"] + cached_tokens * p["cache_read"] + output_tokens * p["output"]) / 1e6


@cache
def _prices_by_id() -> dict:
    return {m["id"]: m for m in MODELS.values()}


def _record(model_id: str, usage) -> None:
    meter = _active_meter.get()
    if meter is None or usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    meter.calls += 1
    meter.input_tokens += usage.prompt_tokens
    meter.cached_tokens += cached
    meter.output_tokens += usage.completion_tokens
    meter.cost += price(model_id, usage.prompt_tokens, cached, usage.completion_tokens)


# ---------------------------------------------------------------- our calls


TIMEOUT = 3600  # seconds per call: give slow, long-thinking calls every chance to finish


def reasoning_body(config: str) -> dict:
    """Request body that sets the reasoning effort, in OpenRouter's `reasoning` form.

    Used for our own calls and PageIndex's, so every call sets effort the same way.
    """
    _, effort = resolve(config)
    return {} if effort == "default" else {"reasoning": {"effort": effort}}


@cache
def _client() -> OpenAI:
    return OpenAI(base_url=base_url(), api_key=api_key(), timeout=TIMEOUT)


def _retry(call, retries: int = 6):
    for attempt in range(retries):
        try:
            return call()
        except (RateLimitError, APIError) as e:
            if attempt == retries - 1 or getattr(e, "status_code", 500) in (400, 401, 403, 404):
                raise
            time.sleep(2**attempt)


def _chat(config: str, messages: list[dict], **kwargs):
    """One metered chat completion. Returns the assistant message object."""
    model_id, _ = resolve(config)
    if body := reasoning_body(config):
        kwargs["extra_body"] = body
    response = _retry(lambda: _client().chat.completions.create(model=model_id, messages=messages, **kwargs))
    _record(model_id, response.usage)
    return response.choices[0].message


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


# ---------------------------------------------------------------- PageIndex


def pageindex_client(storage_path: Path):
    """A local-mode PageIndexClient whose index and chat calls go through LLM Foundry."""
    from pageindex import PageIndexClient

    _meter_litellm()
    backend = {"api_key": api_key(), "api_base": base_url()}
    index = pageindex_chat_args(CONFIG["index_model"])
    return PageIndexClient(
        index_model=index["model"],
        chat_model=index["model"],  # every chat() call passes its own model
        # index_backend is passed verbatim to every indexing litellm call
        index_backend=backend | {"timeout": TIMEOUT, "extra_body": index["extra_body"]},
        chat_backend=backend,
        storage_path=str(storage_path),
        summary_concurrency=16,  # PageIndex default is 64 parallel calls per document; be gentler on LLM Foundry
    )


def pageindex_chat_args(config: str) -> dict:
    """`model` and `extra_body` (reasoning effort) arguments for PageIndexClient.chat()."""
    model_id, _ = resolve(config)
    return {"model": "openrouter/" + model_id, "extra_body": reasoning_body(config)}


@cache
def _meter_litellm() -> None:
    """Wrap litellm.(a)completion so PageIndex's internal calls are metered."""
    import logging

    import litellm

    # LiteLLM's background logging tasks are cancelled at every asyncio.run() exit; asyncio
    # reports each one as "Task was destroyed but it is pending". Harmless, so silence it.
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    completion, acompletion = litellm.completion, litellm.acompletion

    def model_id(kwargs) -> str:
        return kwargs["model"].removeprefix("openrouter/")

    def metered_completion(*args, **kwargs):
        response = completion(*args, **kwargs)
        _record(model_id(kwargs), getattr(response, "usage", None))
        return response

    async def metered_acompletion(*args, **kwargs):
        response = await acompletion(*args, **kwargs)
        _record(model_id(kwargs), getattr(response, "usage", None))
        return response

    litellm.completion, litellm.acompletion = metered_completion, metered_acompletion
