from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI

from src.config import KB_CFG, LLM_CFG
from src.embedder import load_embedder, embed_texts
from src.vector_store import load_faiss_index, load_metadata


class Retriever:
    """
    Semantic retriever over the PPE / Safety knowledge base.

    Usage
    -----
    retriever = Retriever()
    answer = retriever.answer("worker without helmet in restricted area")
    print(answer)
    """

    def __init__(
        self,
        index_path: Path = KB_CFG.faiss_index,
        meta_path: Path = KB_CFG.metadata_json,
        model_name: str = KB_CFG.embedding_model,
        min_score: float = KB_CFG.min_score,
    ):
        self._index = load_faiss_index(index_path)
        self._records = load_metadata(meta_path)
        self._embedder = load_embedder(model_name)
        self._min_score = min_score
        self._client = OpenAI(api_key=LLM_CFG.api_key or None)

    # ------------------------------------------------------------------
    def query(
        self,
        query_text: str,
        top_k: int = KB_CFG.top_k,
    ) -> List[Dict[str, Any]]:
        """
        Return top-k chunks most similar to *query_text*.

        Returns
        -------
        list of dicts:
            chunk_id, source_file, page_start, page_end, text, score
        """
        if not query_text.strip():
            return []

        vec = embed_texts(self._embedder, [query_text])  # shape (1, dim)
        scores, indices = self._index.search(vec, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue
            if float(score) < self._min_score:
                continue

            rec = self._records[idx]
            results.append(
                {
                    "chunk_id": rec.chunk_id,
                    "source_file": rec.source_file,
                    "page_start": rec.page_start,
                    "page_end": rec.page_end,
                    "text": rec.text,
                    "score": round(float(score), 4),
                }
            )

        return results

    # ------------------------------------------------------------------
    def format_rag_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a readable RAG context block
        to be injected into the LLM prompt.
        """
        if not chunks:
            return "No relevant safety document clauses were found in the knowledge base."

        lines = ["=== RETRIEVED SAFETY DOCUMENT CONTEXT ===\n"]
        for i, ch in enumerate(chunks, 1):
            lines.append(
                f"[Source {i}] Document: {ch['source_file']}  "
                f"| Pages {ch['page_start']}–{ch['page_end']}  "
                f"| Relevance: {ch['score']:.2f}\n"
                f"{ch['text']}\n"
                f"{'-'*60}"
            )
        lines.append("=== END OF CONTEXT ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def answer(self, query_text: str, top_k: int = KB_CFG.top_k) -> str:
        """
        Retrieve context and ask OpenAI for a concise answer.
        """
        chunks = self.query(query_text, top_k=top_k)
        context = self.format_rag_context(chunks)

        prompt = (
            f"User query:\n{query_text}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Write the answer in 5-10 short lines.\n"
            "Use bullet points when listing findings, risks, or recommendations.\n"
            "Keep it concise, practical, and easy to understand.\n"
            "Do not add long explanations or extra filler."
        )

        response = self._client.responses.create(
            model=LLM_CFG.model,
            instructions=LLM_CFG.system_prompt,
            input=prompt,
            max_output_tokens=LLM_CFG.max_tokens,
        )

        return response.output_text.strip()

    # ------------------------------------------------------------------
    @staticmethod
    def violation_query(violation_type: str, zone: str = "", extra: str = "") -> str:
        """
        Build a natural-language query from a structured violation event
        so the retriever can find the most relevant SOP clause.
        """
        base_queries = {
            "PPE_HELMET_MISSING": "mandatory helmet hard hat PPE requirement head protection",
            "PPE_VEST_MISSING": "safety vest high visibility PPE requirement",
            "PPE_MASK_MISSING": "face mask respiratory protection PPE requirement",
            "PPE_GLOVES_MISSING": "gloves hand protection PPE requirement",
            "RESTRICTED_ZONE": "restricted zone unauthorized access prohibited area entry",
            "BLOCKED_EXIT": "emergency exit blocked obstruction evacuation route",
            "BLOCKED_EXTINGUISHER": "fire extinguisher blocked obstructed access firefighting",
            "OVERCROWDING": "maximum occupancy crowd density zone capacity limit",
            "UNSAFE_POSTURE": "ergonomic risk fall prevention posture lifting technique",
            "OTHER": "workplace safety violation SOP compliance",
        }
        q = base_queries.get(violation_type, "workplace safety compliance")
        if zone:
            q += f" {zone}"
        if extra:
            q += f" {extra}"
        return q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retriever and print a concise OpenAI answer.")
    parser.add_argument(
        "query",
        nargs="?",
        default="worker without helmet in restricted area",
        help="Query text to search and answer",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=KB_CFG.top_k,
        help="Number of chunks to retrieve",
    )

    args = parser.parse_args()

    retriever = Retriever()
    answer = retriever.answer(args.query, top_k=args.top_k)
    print(answer)