from pathlib import Path
from typing import Sequence, List
import json
import faiss
from dataclasses import dataclass, asdict

@dataclass
class ChunkRecord:
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    text: str

def save_jsonl(records: Sequence[ChunkRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

def save_metadata(records: Sequence[ChunkRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_metadata(path: Path) -> List[ChunkRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ChunkRecord(**item) for item in raw]

def build_faiss_index(vectors):
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def save_faiss_index(index, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))

def load_faiss_index(path: Path):
    return faiss.read_index(str(path))