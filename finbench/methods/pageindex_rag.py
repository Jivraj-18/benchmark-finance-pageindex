"""PageIndex (open-source SDK, local mode, Flash indexing).

`prepare()` builds one tree per filing with the index model in config/models.yaml
and records what that cost. `answer()` lets the chosen model search the tree.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import cache

from tqdm import tqdm

from ..data import CACHE, pdf_path
from ..llm import Meter, pageindex_chat_args, pageindex_client
from ..prompts import ANSWER_INSTRUCTIONS

STORAGE = CACHE / "pageindex"
DOC_IDS = STORAGE / "doc_ids.json"  # doc_name -> {"doc_id", "index_cost", ...}
_lock = threading.Lock()


@cache
def _client():
    return pageindex_client(STORAGE)


def _indexed() -> dict:
    return json.loads(DOC_IDS.read_text()) if DOC_IDS.exists() else {}


def prepare(doc_names: list[str], workers: int = 4) -> dict:
    """Index each filing once, `workers` at a time. Returns {doc_name: index record}."""
    indexed = _indexed()

    def index(doc: str) -> None:
        try:
            with Meter() as usage:
                doc_id = _client().submit_document(str(pdf_path(doc)))["doc_id"]
        except Exception as e:  # its questions are recorded as errors and retried on the next run
            print(f"PageIndex failed to index {doc}: {e}")
            return
        with _lock:  # saved after each filing, so an interrupted run keeps its progress
            indexed[doc] = {"doc_id": doc_id, **usage.to_dict()}
            STORAGE.mkdir(parents=True, exist_ok=True)
            DOC_IDS.write_text(json.dumps(indexed, indent=1))

    todo = [doc for doc in doc_names if doc not in indexed]
    with ThreadPoolExecutor(workers) as pool:
        list(tqdm(pool.map(index, todo), total=len(todo), desc="PageIndex trees"))
    return {doc: indexed[doc] for doc in doc_names if doc in indexed}


def answer(question: dict, config: str) -> dict:
    doc_id = _indexed()[question["doc_name"]]["doc_id"]
    response = _client().chat(
        question["question"],
        doc_id=doc_id,
        protocol="chat_completions",
        instructions=ANSWER_INSTRUCTIONS,
        **pageindex_chat_args(config),
    )
    return {"answer": response["choices"][0]["message"]["content"] or ""}
