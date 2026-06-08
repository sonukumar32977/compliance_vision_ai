"""
Compliance Vision AI — LLM Report Generator
Assembles a structured prompt from a violation event + RAG context,
calls the OpenAI API, and returns a parsed ViolationReport.
"""

from __future__ import annotations
import json
import uuid
import datetime
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config import LLM_CFG, LLMConfig
from src.violation_logic import ViolationEvent
from src.retriever import Retriever


# ──────────────────────────────────────────────────────────────
# Output data model
# ──────────────────────────────────────────────────────────────

@dataclass
class CitedRule:
    document:        Optional[str] = None
    section:         Optional[str] = None
    page:            Optional[int] = None
    clause_summary:  Optional[str] = None


@dataclass
class ViolationReport:
    violation_id:         str
    violation_type:       str
    certainty:            str
    confidence_score:     float
    timestamp:            str
    zone:                 str
    camera_id:            str
    evidence_frame:       str
    cited_rule:           CitedRule
    risk_level:           str
    risk_justification:   str
    recommendation:       str
    remediation_timeline: str
    narrative:            str
    raw_rag_context:      str = ""
    generated_at:         str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ──────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────

def _build_user_prompt(event: ViolationEvent, rag_context: str) -> str:
    """
    Builds the dynamic user prompt injected at inference time.
    Mirrors Section 5 of the Compliance Vision AI Master Prompt specification.
    """
    det_summary = "\n".join(
        f"  - {d['class']} (conf: {d['confidence']:.2f}) bbox: {d['bbox']}"
        for d in event.detections
    ) or "  - No specific objects listed (violation inferred from absence of PPE)"

    return f"""
=== VIOLATION DETECTION REPORT ===
Violation ID      : {event.violation_id}
Violation Type    : {event.violation_type}
Timestamp         : {event.timestamp}
Frame ID          : {event.frame_id}
Detection Conf.   : {event.confidence:.2f}
Zone              : {event.zone_id}
Camera            : {event.camera_id}
Description       : {event.description}

=== DETECTED OBJECTS IN FRAME ===
{det_summary}

=== RETRIEVED SAFETY DOCUMENT CONTEXT (RAG) ===
{rag_context}

=== YOUR TASK ===
Using the violation data and RAG context above, produce:

1. A JSON object matching this exact schema:
{{
  "violation_id"        : "{event.violation_id}",
  "violation_type"      : "<one of the defined types>",
  "certainty"           : "Confirmed | Probable | Possible",
  "confidence_score"    : <float 0.0–1.0>,
  "timestamp"           : "{event.timestamp}",
  "zone"                : "{event.zone_id}",
  "camera_id"           : "{event.camera_id}",
  "evidence_frame"      : "{event.screenshot_path}",
  "cited_rule"          : {{
      "document"        : "<full title of source document or null>",
      "section"         : "<section number and heading or null>",
      "page"            : <integer or null>,
      "clause_summary"  : "<one sentence or null>"
  }},
  "risk_level"          : "Critical | High | Medium | Low",
  "risk_justification"  : "<one sentence>",
  "recommendation"      : "<specific corrective action>",
  "remediation_timeline": "Immediate | Within 24 hours | Within 1 week",
  "narrative"           : "<2-3 sentences for safety officer>"
}}

2. Then on a new line write: NARRATIVE: followed by the same narrative text.

Return ONLY valid JSON + the NARRATIVE line. No extra commentary.
""".strip()


# ──────────────────────────────────────────────────────────────
# Report generator
# ──────────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Calls OpenAI to generate a structured violation report from a
    ViolationEvent + RAG context.

    Parameters
    ----------
    retriever   : Retriever instance (for semantic search)
    llm_config  : LLMConfig (API key, model, temperature, …)
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_config: LLMConfig = LLM_CFG,
    ):
        self.retriever  = retriever
        self.cfg        = llm_config
        self._client    = OpenAI(api_key=llm_config.api_key)

    # ------------------------------------------------------------------
    def generate(self, event: ViolationEvent) -> ViolationReport:
        """
        Full pipeline:
          1. Build retrieval query from violation type
          2. Retrieve top-k RAG chunks
          3. Build prompt
          4. Call LLM
          5. Parse response → ViolationReport
        """
        # Step 1 — RAG retrieval
        query      = Retriever.violation_query(event.violation_type, event.zone_id)
        chunks     = self.retriever.query(query)
        rag_ctx    = self.retriever.format_rag_context(chunks)

        # Step 2 — LLM call
        user_prompt = _build_user_prompt(event, rag_ctx)
        raw_response = self._call_llm(user_prompt)

        # Step 3 — Parse
        report = self._parse_response(raw_response, event, rag_ctx)
        return report

    # ------------------------------------------------------------------
    def generate_batch(self, events: List[ViolationEvent]) -> List[ViolationReport]:
        """Generate reports for a list of violation events."""
        reports = []
        for ev in events:
            try:
                reports.append(self.generate(ev))
            except Exception as e:
                print(f"[ERROR] Report generation failed for {ev.violation_id}: {e}")
                reports.append(self._fallback_report(ev, str(e)))
        return reports

    # ------------------------------------------------------------------
    def _call_llm(self, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model       = self.cfg.model,
            temperature = self.cfg.temperature,
            max_tokens  = self.cfg.max_tokens,
            timeout     = self.cfg.timeout,
            messages=[
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(
        raw: str,
        event: ViolationEvent,
        rag_ctx: str,
    ) -> ViolationReport:
        """
        Extract the JSON block from the LLM response and map it to
        a ViolationReport dataclass.  Falls back gracefully on parse errors.
        """
        # Try to isolate the JSON block
        json_str  = raw
        narrative = ""

        if "NARRATIVE:" in raw:
            parts     = raw.split("NARRATIVE:", 1)
            json_str  = parts[0].strip()
            narrative = parts[1].strip()

        # Strip markdown code fences if present
        if json_str.startswith("```"):
            lines    = json_str.split("\n")
            json_str = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Attempt to extract first {...} block
            import re
            m = re.search(r"\{.*\}", json_str, re.DOTALL)
            data = json.loads(m.group()) if m else {}

        cr_raw = data.get("cited_rule") or {}
        cited  = CitedRule(
            document       = cr_raw.get("document"),
            section        = cr_raw.get("section"),
            page           = cr_raw.get("page"),
            clause_summary = cr_raw.get("clause_summary"),
        )

        return ViolationReport(
            violation_id         = data.get("violation_id", event.violation_id),
            violation_type       = data.get("violation_type", event.violation_type),
            certainty            = data.get("certainty", "Probable"),
            confidence_score     = float(data.get("confidence_score", event.confidence)),
            timestamp            = data.get("timestamp", event.timestamp),
            zone                 = data.get("zone", event.zone_id),
            camera_id            = data.get("camera_id", event.camera_id),
            evidence_frame       = data.get("evidence_frame", event.screenshot_path),
            cited_rule           = cited,
            risk_level           = data.get("risk_level", "High"),
            risk_justification   = data.get("risk_justification", ""),
            recommendation       = data.get("recommendation", ""),
            remediation_timeline = data.get("remediation_timeline", "Immediate"),
            narrative            = data.get("narrative", narrative),
            raw_rag_context      = rag_ctx,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _fallback_report(event: ViolationEvent, error: str) -> ViolationReport:
        return ViolationReport(
            violation_id         = event.violation_id,
            violation_type       = event.violation_type,
            certainty            = "Possible",
            confidence_score     = event.confidence,
            timestamp            = event.timestamp,
            zone                 = event.zone_id,
            camera_id            = event.camera_id,
            evidence_frame       = event.screenshot_path,
            cited_rule           = CitedRule(),
            risk_level           = "High",
            risk_justification   = "Report generation encountered an error; manual review required.",
            recommendation       = "Manually inspect violation and consult site safety officer.",
            remediation_timeline = "Immediate",
            narrative            = f"Automated report generation failed: {error}. Manual review is required.",
        )
