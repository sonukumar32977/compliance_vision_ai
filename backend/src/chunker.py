import re
from typing import List

def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=(?:[\(\"'\[]|[A-Z0-9]))", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, chunk_size_words: int = 220, overlap_words: int = 40) -> List[str]:
    if not text.strip():
        return []

    sentences = sentence_split(text)
    chunks = []
    current = []
    current_words = 0

    def wc(s: str) -> int:
        return len(s.split())

    for sent in sentences:
        sent_words = wc(sent)
        if current and current_words + sent_words > chunk_size_words:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)

            if overlap_words > 0 and chunk:
                words = chunk.split()
                tail = words[-overlap_words:] if len(words) > overlap_words else words[:]
                current = [" ".join(tail)] if tail else []
                current_words = len(tail)
            else:
                current = []
                current_words = 0

        current.append(sent)
        current_words += sent_words

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    final_chunks = []
    seen = set()
    for ch in chunks:
        norm = re.sub(r"\s+", " ", ch.lower()).strip()
        if len(norm.split()) < 30:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        final_chunks.append(ch)

    return final_chunks