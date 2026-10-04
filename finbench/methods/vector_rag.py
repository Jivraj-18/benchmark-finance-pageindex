"""Classic vector RAG: chunk the filing, embed the chunks, retrieve the closest to the question.

`prepare()` splits each page into overlapping chunks, embeds them with `embedding_model`
(config/models.yaml) and caches the vectors. `answer()` embeds the question, takes the
TOP_K chunks by cosine similarity, and asks the model to answer from those excerpts only.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from ..data import CACHE, page_texts
from ..llm import Meter, complete, embed
from ..prompts import ANSWER_INSTRUCTIONS

STORE = CACHE / "vectors"
CHUNK_CHARS = 2000  # ~500 tokens
OVERLAP_CHARS = 300
TOP_K = 10


def chunks(doc_name: str) -> list[dict]:
    """Overlapping character windows within each page, so every chunk has one page number."""
    out = []
    for page, text in enumerate(page_texts(doc_name), start=1):
        text = " ".join(text.split())
        for start in range(0, max(len(text), 1), CHUNK_CHARS - OVERLAP_CHARS):
            piece = text[start : start + CHUNK_CHARS]
            if piece.strip():
                out.append({"page": page, "text": piece})
    return out


def prepare(doc_names: list[str], workers: int = 4) -> dict:
    """Embed each filing once. Returns {doc_name: embedding cost record}."""
    STORE.mkdir(parents=True, exist_ok=True)

    def index(doc: str) -> tuple[str, dict]:
        meta = STORE / f"{doc}.json"
        if meta.exists():
            return doc, json.loads(meta.read_text())["usage"]
        pieces = chunks(doc)
        with Meter() as usage:
            vectors = np.array(embed([p["text"] for p in pieces]), dtype=np.float32)
        np.save(STORE / f"{doc}.npy", vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
        meta.write_text(json.dumps({"chunks": pieces, "usage": usage.to_dict()}))
        return doc, usage.to_dict()

    with ThreadPoolExecutor(workers) as pool:
        return dict(tqdm(pool.map(index, doc_names), total=len(doc_names), desc="Embeddings"))


def retrieve(question: str, doc_name: str) -> list[dict]:
    """The TOP_K chunks closest to the question, best first, with their scores."""
    vectors = np.load(STORE / f"{doc_name}.npy")
    pieces = json.loads((STORE / f"{doc_name}.json").read_text())["chunks"]
    query = np.array(embed([question])[0], dtype=np.float32)
    scores = vectors @ (query / np.linalg.norm(query))
    best = np.argsort(-scores)[:TOP_K]
    return [pieces[i] | {"score": round(float(scores[i]), 4)} for i in best]


def answer(question: dict, config: str) -> dict:
    hits = retrieve(question["question"], question["doc_name"])
    excerpts = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": ANSWER_INSTRUCTIONS},
        {"role": "user", "content": f"Excerpts from the document, most relevant first:\n\n{excerpts}\n\n"
                                    f"Question: {question['question']}"},
    ]
    return {"answer": complete(config, messages), "retrieved": [{"page": h["page"], "score": h["score"]} for h in hits]}
