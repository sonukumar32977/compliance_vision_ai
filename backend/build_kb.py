"""
Compliance Vision AI — Knowledge Base Builder
CLI: scans data/raw for PDFs, extracts, chunks, embeds → FAISS index.

Usage
-----
    python build_kb.py
    python build_kb.py --input path/to/docs --force
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config       import KB_CFG, DATA_RAW, DATA_PROC
from src.extractor    import extract_pdf_text, list_pdfs
from src.cleaner      import clean_text
from src.chunker      import chunk_text
from src.embedder     import load_embedder, embed_texts
from src.vector_store import (
    ChunkRecord, save_jsonl, save_metadata,
    build_faiss_index, save_faiss_index,
)


def build_knowledge_base(input_dir=DATA_RAW, output_dir=DATA_PROC, force=False):
    index_path = output_dir / "index.faiss"
    meta_path  = output_dir / "metadata.json"
    jsonl_path = output_dir / "chunks.jsonl"

    if index_path.exists() and not force:
        print(f"[INFO] Index exists at {index_path}. Use --force to rebuild.")
        return

    # 1 – Discover PDFs
    print(f"\n[1/6] Scanning: {input_dir}")
    try:
        pdfs = list_pdfs(input_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}\n       Add PDF(s) to data/raw/ and re-run.")
        sys.exit(1)
    for p in pdfs:
        print(f"       • {p.name}")

    # 2 – Extract + clean + chunk
    print("\n[2/6] Extracting text and chunking …")
    all_records, chunk_idx = [], 0

    for pdf_path in pdfs:
        pages = extract_pdf_text(pdf_path)
        full_text = " ".join(clean_text(pg["text"]) for pg in pages)
        chunks = chunk_text(full_text, KB_CFG.chunk_size_words, KB_CFG.overlap_words)
        total_pages = len(pages)
        print(f"       {pdf_path.name}: {len(pages)} pages -> {len(chunks)} chunks")

        for i, ch in enumerate(chunks):
            frac = i / max(len(chunks) - 1, 1)
            p_start = max(1, int(frac * total_pages))
            p_end   = min(total_pages, p_start + 1)
            all_records.append(ChunkRecord(
                chunk_id    = f"{pdf_path.stem}__chunk_{chunk_idx:04d}",
                source_file = pdf_path.name,
                page_start  = p_start,
                page_end    = p_end,
                text        = ch,
            ))
            chunk_idx += 1

    print(f"       Total chunks: {len(all_records)}")
    if not all_records:
        print("[ERROR] No chunks produced. Ensure PDFs are text-searchable.")
        sys.exit(1)

    # 3 – Save JSONL
    print("\n[3/6] Saving chunks.jsonl …")
    save_jsonl(all_records, jsonl_path)

    # 4 – Embed
    print(f"\n[4/6] Embedding with {KB_CFG.embedding_model} …")
    t0 = time.time()
    embedder = load_embedder(KB_CFG.embedding_model)
    vectors  = embed_texts(embedder, [r.text for r in all_records])
    print(f"       Done in {time.time()-t0:.1f}s  shape={vectors.shape}")

    # 5 – FAISS index
    print("\n[5/6] Building + saving FAISS index …")
    save_faiss_index(build_faiss_index(vectors), index_path)
    print(f"       -> {index_path}")

    # 6 – Metadata
    print("\n[6/6] Saving metadata.json …")
    save_metadata(all_records, meta_path)
    print(f"       -> {meta_path}")

    print(f"\n[OK] Knowledge base ready | docs={len(pdfs)} chunks={len(all_records)} dim={vectors.shape[1]}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compliance Vision AI — Build knowledge base")
    ap.add_argument("--input",  "-i", type=Path, default=DATA_RAW)
    ap.add_argument("--output", "-o", type=Path, default=DATA_PROC)
    ap.add_argument("--force",  "-f", action="store_true")
    a = ap.parse_args()
    build_knowledge_base(a.input, a.output, a.force)
