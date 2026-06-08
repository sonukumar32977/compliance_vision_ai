from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.config import LLM_CFG


@dataclass(frozen=True)
class OpenAIEmbedder:
    """Wrapper around the OpenAI embeddings API."""

    client: OpenAI
    model: str


def load_embedder(model_name: str = "text-embedding-3-small") -> OpenAIEmbedder:
    if not LLM_CFG.api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to .env before building or querying the knowledge base."
        )
    return OpenAIEmbedder(
        client=OpenAI(api_key=LLM_CFG.api_key),
        model=model_name,
    )


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so FAISS IndexFlatIP behaves like cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def embed_texts(
    embedder: OpenAIEmbedder,
    texts: Sequence[str],
    batch_size: int = 100,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts via OpenAI and return a float32 matrix (n, dim).
    Empty strings are replaced with a single space (API rejects empty input).
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    cleaned = [t if t.strip() else " " for t in texts]
    all_vectors: list[list[float]] = []

    batches = range(0, len(cleaned), batch_size)
    iterator = batches
    if show_progress and len(cleaned) > batch_size:
        iterator = tqdm(
            list(batches),
            desc=f"OpenAI embeddings ({embedder.model})",
            unit="batch",
        )

    for start in iterator:
        batch = cleaned[start : start + batch_size]
        response = embedder.client.embeddings.create(
            input=batch,
            model=embedder.model,
        )
        # API returns embeddings in input order
        ordered = sorted(response.data, key=lambda x: x.index)
        all_vectors.extend([item.embedding for item in ordered])

    vectors = np.asarray(all_vectors, dtype="float32")
    return _normalize(vectors)
