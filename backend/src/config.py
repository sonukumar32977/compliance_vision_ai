"""
Compliance Vision AI — Central Configuration
All tuneable parameters for the entire pipeline live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────
# Directory layout (all relative to the project root)
# ──────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
PROJECT_ROOT  = ROOT.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(ROOT / ".env", override=True)
DATA_RAW      = ROOT / "data" / "raw"
DATA_PROC     = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
OUTPUTS_DIR   = ROOT / "outputs"
SCREENSHOTS   = OUTPUTS_DIR / "screenshots"
REPORTS_DIR   = OUTPUTS_DIR / "reports"

# Ensure folders exist
for _d in [DATA_RAW, DATA_PROC, MODELS_DIR, OUTPUTS_DIR, SCREENSHOTS, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Knowledge-Base / RAG settings
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class KBConfig:
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    chunk_size_words: int = 220
    overlap_words: int    = 40
    top_k: int            = 3          # chunks retrieved per query
    min_score: float      = 0.20       # cosine similarity threshold

    chunks_jsonl: Path    = DATA_PROC / "chunks.jsonl"
    metadata_json: Path   = DATA_PROC / "metadata.json"
    faiss_index: Path     = DATA_PROC / "index.faiss"


# ──────────────────────────────────────────────────────────────
# YOLOv8 / Video settings
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DetectionConfig:
    model_path: Path       = MODELS_DIR / "best.pt"
    confidence: float      = 0.40      # minimum detection confidence
    iou_threshold: float   = 0.45
    frame_rate: int        = 1         # frames to sample per second (0 = every frame)
    input_size: int        = 640       # YOLO input square size

    # Class names the model is trained to detect
    classes: tuple = (
        "person",
        "helmet",
        "safety_vest",
        "face_mask",
        "gloves",
        "fire_extinguisher",
        "emergency_exit",
        "restricted_zone_marker",
    )


# ──────────────────────────────────────────────────────────────
# LLM / OpenAI settings
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LLMConfig:
    api_key: str        = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature: float  = 0.2
    max_tokens: int     = 1200
    timeout: int        = 60

    system_prompt: str = (
        "You are Compliance Vision AI, an expert workplace-safety compliance analyst. "
        "Your role is to analyse PPE and safety violation evidence detected by computer-vision "
        "systems, retrieve the exact regulatory clause or SOP section that was violated, and "
        "produce a clear, actionable violation report.\n\n"
        "STRICT RULES:\n"
        "1. Always ground every cited_rule in the RAG context provided. Never fabricate citations.\n"
        "2. Return a valid JSON object matching the schema exactly, followed by a plain-language "
        "narrative paragraph beginning with 'NARRATIVE:'.\n"
        "3. risk_level must be one of: Critical | High | Medium | Low.\n"
        "4. certainty must be one of: Confirmed | Probable | Possible.\n"
        "5. remediation_timeline must be one of: Immediate | Within 24 hours | Within 1 week.\n"
        "6. If no relevant RAG context is found, set cited_rule fields to null and note it in "
        "the narrative.\n"
        "7. Write recommendations as imperative sentences (e.g., 'Immediately issue helmet ...').\n"
        "8. Keep narrative to 2-3 sentences targeted at a safety officer on the floor."
    )


# ──────────────────────────────────────────────────────────────
# Violation catalogue
# ──────────────────────────────────────────────────────────────
VIOLATION_TYPES = {
    "PPE_HELMET_MISSING":       "Person detected without a helmet in a mandatory hard-hat zone.",
    "PPE_VEST_MISSING":         "Person detected without a high-visibility safety vest.",
    "PPE_MASK_MISSING":         "Person detected without a face mask in a required zone.",
    "PPE_GLOVES_MISSING":       "Person detected without gloves in a hand-protection zone.",
    "RESTRICTED_ZONE":          "Unauthorised personnel detected inside a restricted area.",
    "BLOCKED_EXIT":             "Emergency exit obstructed by a person or object cluster.",
    "BLOCKED_EXTINGUISHER":     "Fire extinguisher occluded or blocked by another object.",
    "OVERCROWDING":             "Person count in zone exceeds the configured safe threshold.",
    "UNSAFE_POSTURE":           "Worker posture indicates a potential ergonomic or fall risk.",
    "OTHER":                    "General safety violation not covered by a specific category.",
}

# ──────────────────────────────────────────────────────────────
# Singleton accessors
# ──────────────────────────────────────────────────────────────
KB_CFG        = KBConfig()
DETECT_CFG    = DetectionConfig()
LLM_CFG       = LLMConfig()
