import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Fix null bytes
    text = text.replace("\x00", " ")

    # Fix hyphenated line-breaks (e.g. "work-\nplace" → "workplace")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\r", "\n")

    # ── Remove PDF dot-leader noise ────────────────────────────
    # Matches patterns like:  . . . . .  or  ......  (3+ dots/spaces)
    text = re.sub(r"(\s*\.\s*){3,}", " ", text)

    # Remove standalone page numbers that appear after a dot trail
    # e.g.  "Section heading . . . . 12"  → "Section heading"
    text = re.sub(r"\s+\d{1,3}\s*$", " ", text, flags=re.MULTILINE)

    # Remove "Page X" / "X of Y" artifacts
    text = re.sub(r"\bPage\s+\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s+of\s+\d+\b", " ", text, flags=re.IGNORECASE)

    # Remove lines that are ONLY whitespace + digits (page number lines)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", " ", text)

    # Collapse multiple spaces / newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text