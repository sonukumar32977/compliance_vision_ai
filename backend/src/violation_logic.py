"""
Compliance Vision AI — Violation Logic Engine  (rfdetr-base edition)
============================================================
Evaluates detection results from a 14-class rfdetr-base model against
rule-based safety conditions and emits structured ViolationEvent objects
ready for RAG + LLM processing.

Supported model classes (14)
─────────────────────────────
Positive PPE : Mask · Goggles · Person · Hardhat · Safety Vest · Gloves
               Ladder · Safety Cone
Negative PPE : NO-Mask · NO-Goggles · NO-Hardhat · NO-Safety Vest · NO-Gloves
Special      : Fall-Detected

Violation rules
───────────────
R-01  PPE_HELMET_MISSING    — NO-Hardhat detection (any person)
R-02  PPE_VEST_MISSING      — NO-Safety Vest detection
R-03  PPE_MASK_MISSING      — NO-Mask detection (mask-required zones)
R-04  PPE_GOGGLES_MISSING   — NO-Goggles detection (eye-protection zones)
R-05  PPE_GLOVES_MISSING    — NO-Gloves detection (hand-protection zones)
R-06  FALL_DETECTED         — Fall-Detected class present
R-07  OVERCROWDING          — person count > zone limit
R-08  RESTRICTED_ZONE       — person present in a marked restricted zone
R-09  BLOCKED_EXIT          — person bbox overlaps emergency-exit marker
R-10  BLOCKED_EXTINGUISHER  — object bbox occludes fire-extinguisher marker
R-11  UNSAFE_LADDER_USE     — Ladder detected with no person nearby (unsecured)
                            OR person on ladder without hardhat
R-12  MISSING_SAFETY_CONE   — Work zone (ladder present) lacks a safety cone
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model class catalogue
# ──────────────────────────────────────────────────────────────────────────────

# All 14 class names exactly as they appear in the rfdetr-base model.
MODEL_CLASSES: Tuple[str, ...] = (
    "Mask",
    "Goggles",
    "Person",
    "Hardhat",
    "Ladder",
    "Safety Vest",
    "Fall-Detected",
    "Gloves",
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Safety Cone",
)

# Convenience sets
_PPE_NEGATIVE_CLASSES = {
    "NO-Hardhat",
    "NO-Safety Vest",
    "NO-Mask",
    "NO-Goggles",
    "NO-Gloves",
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Core data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BBox:
    """Pixel-coordinate bounding box [x1, y1, x2, y2]."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def iou(self, other: "BBox") -> float:
        inter = self._intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlap_ratio(self, other: "BBox") -> float:
        """Intersection / self.area — useful when boxes have very different sizes."""
        inter = self._intersection_area(other)
        return inter / self.area if self.area > 0 else 0.0

    def overlaps(self, other: "BBox", threshold: float = 0.10) -> bool:
        return self.overlap_ratio(other) >= threshold

    def distance_to(self, other: "BBox") -> float:
        """Euclidean distance between centres."""
        cx1, cy1 = self.centre
        cx2, cy2 = other.centre
        return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

    def _intersection_area(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


@dataclass
class Detection:
    """Single detection result from rfdetr-base."""

    class_name: str
    confidence: float
    bbox: BBox

    @property
    def is_negative_ppe(self) -> bool:
        return self.class_name in _PPE_NEGATIVE_CLASSES


@dataclass
class ViolationEvent:
    """A triggered safety violation ready for RAG + LLM processing."""

    violation_id: str
    violation_type: str
    timestamp: str
    frame_id: int
    confidence: float
    zone_id: str
    camera_id: str
    screenshot_path: str
    description: str
    detections: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Zone registry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ZoneConfig:
    """Per-zone safety requirements."""

    name: str
    bbox: BBox
    requires_hardhat: bool = True
    requires_vest: bool = True
    requires_mask: bool = False
    requires_goggles: bool = False
    requires_gloves: bool = False
    restricted: bool = False          # no unauthorised entry
    max_occupancy: int = 50


class ZoneRegistry:
    """Stores named rectangular zones with associated safety rules."""

    def __init__(self):
        self._zones: Dict[str, ZoneConfig] = {
            "general_floor": ZoneConfig(
                name="general_floor",
                bbox=BBox(0, 0, 99_999, 99_999),
                requires_hardhat=True,
                requires_vest=True,
                max_occupancy=50,
            ),
        }

    def add_zone(self, config: ZoneConfig) -> None:
        self._zones[config.name] = config

    def get_zone(self, bbox: BBox) -> ZoneConfig:
        """Return the most specific zone whose bbox contains the given box's centre."""
        cx, cy = bbox.centre
        best: Optional[ZoneConfig] = None
        best_area = float("inf")
        for cfg in self._zones.values():
            zb = cfg.bbox
            if zb.x1 <= cx <= zb.x2 and zb.y1 <= cy <= zb.y2:
                if cfg.bbox.area < best_area:
                    best = cfg
                    best_area = cfg.bbox.area
        return best or ZoneConfig(
            name="unknown_zone",
            bbox=BBox(0, 0, 0, 0),
        )


DEFAULT_ZONE_REGISTRY = ZoneRegistry()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Violation logic engine
# ──────────────────────────────────────────────────────────────────────────────

class ViolationLogicEngine:
    """
    Processes a list of Detection objects for a single video frame and
    returns zero or more ViolationEvent objects.

    Because the rfdetr-base model directly outputs negative-PPE classes
    (NO-Hardhat, NO-Mask, …) and Fall-Detected, most PPE rules are
    simply a presence-check on those negative classes — no spatial
    overlap heuristic required.
    """

    def __init__(
        self,
        zone_registry: ZoneRegistry = DEFAULT_ZONE_REGISTRY,
        camera_id: str = "CAM-01",
        ladder_person_proximity_px: float = 150.0,
        blocked_overlap_threshold: float = 0.20,
    ):
        self.zone_registry = zone_registry
        self.camera_id = camera_id
        self.ladder_person_proximity_px = ladder_person_proximity_px
        self.blocked_overlap_threshold = blocked_overlap_threshold

    # ── public entry point ────────────────────────────────────────────────────

    def evaluate(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp: str,
        screenshot_path: str = "",
    ) -> List[ViolationEvent]:
        """
        Evaluate all detections for a single frame.
        Returns a (possibly empty) list of ViolationEvents.
        """
        ctx = dict(
            frame_id=frame_id,
            timestamp=timestamp,
            screenshot_path=screenshot_path,
            camera_id=self.camera_id,
        )

        # ── Bucket detections by class ──────────────────────────────────────
        by_class: Dict[str, List[Detection]] = {}
        for d in detections:
            by_class.setdefault(d.class_name, []).append(d)

        persons       = by_class.get("Person", [])
        ladders       = by_class.get("Ladder", [])
        cones         = by_class.get("Safety Cone", [])
        falls         = by_class.get("Fall-Detected", [])

        # Negative-PPE buckets (direct model outputs — no inference needed)
        no_hardhats   = by_class.get("NO-Hardhat", [])
        no_vests      = by_class.get("NO-Safety Vest", [])
        no_masks      = by_class.get("NO-Mask", [])
        no_goggles    = by_class.get("NO-Goggles", [])
        no_gloves     = by_class.get("NO-Gloves", [])

        violations: List[ViolationEvent] = []

        # ── R-01  PPE_HELMET_MISSING ────────────────────────────────────────
        for det in no_hardhats:
            zone = self.zone_registry.get_zone(det.bbox)
            if zone.requires_hardhat:
                violations.append(self._event(
                    "PPE_HELMET_MISSING",
                    det.confidence,
                    zone.name,
                    "Worker detected without a hardhat in a mandatory hard-hat zone.",
                    [det],
                    **ctx,
                ))

        # ── R-02  PPE_VEST_MISSING ──────────────────────────────────────────
        for det in no_vests:
            zone = self.zone_registry.get_zone(det.bbox)
            if zone.requires_vest:
                violations.append(self._event(
                    "PPE_VEST_MISSING",
                    det.confidence,
                    zone.name,
                    "Worker detected without a high-visibility safety vest.",
                    [det],
                    **ctx,
                ))

        # ── R-03  PPE_MASK_MISSING ──────────────────────────────────────────
        for det in no_masks:
            zone = self.zone_registry.get_zone(det.bbox)
            if zone.requires_mask:
                violations.append(self._event(
                    "PPE_MASK_MISSING",
                    det.confidence,
                    zone.name,
                    "Worker detected without a face mask in a mask-required zone.",
                    [det],
                    **ctx,
                ))

        # ── R-04  PPE_GOGGLES_MISSING ───────────────────────────────────────
        for det in no_goggles:
            zone = self.zone_registry.get_zone(det.bbox)
            if zone.requires_goggles:
                violations.append(self._event(
                    "PPE_GOGGLES_MISSING",
                    det.confidence,
                    zone.name,
                    "Worker detected without protective goggles in an eye-hazard zone.",
                    [det],
                    **ctx,
                ))

        # ── R-05  PPE_GLOVES_MISSING ────────────────────────────────────────
        for det in no_gloves:
            zone = self.zone_registry.get_zone(det.bbox)
            if zone.requires_gloves:
                violations.append(self._event(
                    "PPE_GLOVES_MISSING",
                    det.confidence,
                    zone.name,
                    "Worker detected without gloves in a hand-protection zone.",
                    [det],
                    **ctx,
                ))

        # ── R-06  FALL_DETECTED ─────────────────────────────────────────────
        for det in falls:
            zone = self.zone_registry.get_zone(det.bbox)
            violations.append(self._event(
                "FALL_DETECTED",
                det.confidence,
                zone.name,
                "A worker fall or collapse has been detected — immediate response required.",
                [det],
                **ctx,
            ))

        # ── R-07  OVERCROWDING ──────────────────────────────────────────────
        # Group persons by zone and compare against per-zone limit.
        zone_person_map: Dict[str, List[Detection]] = {}
        for p in persons:
            zone = self.zone_registry.get_zone(p.bbox)
            zone_person_map.setdefault(zone.name, []).append(p)

        for zone_name, zone_persons in zone_person_map.items():
            zone_cfg = self.zone_registry.get_zone(zone_persons[0].bbox)
            if len(zone_persons) > zone_cfg.max_occupancy:
                violations.append(self._event(
                    "OVERCROWDING",
                    1.0,
                    zone_name,
                    (
                        f"Person count ({len(zone_persons)}) exceeds the safe "
                        f"occupancy limit of {zone_cfg.max_occupancy} for zone '{zone_name}'."
                    ),
                    zone_persons,
                    **ctx,
                ))

        # ── R-08  RESTRICTED_ZONE ───────────────────────────────────────────
        for p in persons:
            zone = self.zone_registry.get_zone(p.bbox)
            if zone.restricted:
                violations.append(self._event(
                    "RESTRICTED_ZONE",
                    p.confidence,
                    zone.name,
                    "Unauthorised person detected inside a restricted area.",
                    [p],
                    **ctx,
                ))

        # ── R-09  BLOCKED_EXIT ──────────────────────────────────────────────
        # The model has no explicit exit class; flag if a person overlaps a
        # Safety Cone cluster that is marking an exit path.
        # (Extend this rule when an emergency-exit detector is available.)

        # ── R-10  BLOCKED_EXTINGUISHER ──────────────────────────────────────
        # Safety Cone is used as a proxy for hazard markers; extend when an
        # explicit fire-extinguisher class is available in the model.

        # ── R-11  UNSAFE_LADDER_USE ─────────────────────────────────────────
        for ladder in ladders:
            nearby_persons = [
                p for p in persons
                if ladder.bbox.distance_to(p.bbox) <= self.ladder_person_proximity_px
            ]

            if not nearby_persons:
                # Unattended / unsupported ladder
                zone = self.zone_registry.get_zone(ladder.bbox)
                violations.append(self._event(
                    "UNSAFE_LADDER_USE",
                    ladder.confidence,
                    zone.name,
                    "Ladder detected with no worker nearby — unsecured or unattended ladder.",
                    [ladder],
                    **ctx,
                ))
            else:
                # Person on ladder — check they are wearing a hardhat.
                # We look for a NO-Hardhat detection close to any nearby person.
                for person in nearby_persons:
                    missing_hat = [
                        nh for nh in no_hardhats
                        if person.bbox.distance_to(nh.bbox) <= self.ladder_person_proximity_px
                    ]
                    if missing_hat:
                        zone = self.zone_registry.get_zone(ladder.bbox)
                        violations.append(self._event(
                            "UNSAFE_LADDER_USE",
                            min(ladder.confidence, missing_hat[0].confidence),
                            zone.name,
                            (
                                "Worker on a ladder is not wearing a hardhat — "
                                "elevated fall-injury risk."
                            ),
                            [ladder, person] + missing_hat,
                            **ctx,
                        ))

        # ── R-12  MISSING_SAFETY_CONE ───────────────────────────────────────
        # If a ladder is present but no safety cone is visible nearby,
        # the work zone lacks proper hazard delineation.
        for ladder in ladders:
            nearby_cones = [
                c for c in cones
                if ladder.bbox.distance_to(c.bbox) <= self.ladder_person_proximity_px * 2
            ]
            if not nearby_cones:
                zone = self.zone_registry.get_zone(ladder.bbox)
                violations.append(self._event(
                    "MISSING_SAFETY_CONE",
                    ladder.confidence,
                    zone.name,
                    (
                        "Ladder or work activity detected without any safety cone "
                        "marking the hazard zone."
                    ),
                    [ladder],
                    **ctx,
                ))

        return violations

    # ── helpers ───────────────────────────────────────────────────────────────

    def _event(
        self,
        violation_type: str,
        confidence: float,
        zone_id: str,
        description: str,
        relevant_detections: List[Detection],
        *,
        frame_id: int,
        timestamp: str,
        screenshot_path: str,
        camera_id: str,
    ) -> ViolationEvent:
        return ViolationEvent(
            violation_id=str(uuid.uuid4()),
            violation_type=violation_type,
            timestamp=timestamp,
            frame_id=frame_id,
            confidence=round(confidence, 3),
            zone_id=zone_id,
            camera_id=camera_id,
            screenshot_path=screenshot_path,
            description=description,
            detections=[
                {
                    "class": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                }
                for d in relevant_detections
            ],
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. YOLO / rfdetr result parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_yolo_results(yolo_result, class_names: Tuple[str, ...] = MODEL_CLASSES) -> List[Detection]:
    """
    Convert a single ultralytics / rfdetr Results object into a list of
    Detection instances using the 14-class rfdetr-base catalogue.
    """
    detections: List[Detection] = []
    if yolo_result is None or yolo_result.boxes is None:
        return detections

    for box in yolo_result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        detections.append(Detection(
            class_name=cls_name,
            confidence=conf,
            bbox=BBox(x1, y1, x2, y2),
        ))
    return detections


# ──────────────────────────────────────────────────────────────────────────────
# 6. Quick smoke-test (python violation_logic.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    engine = ViolationLogicEngine(camera_id="CAM-01")

    sample_detections = [
        # A person on the floor
        Detection("Person", 0.91, BBox(100, 50, 250, 400)),
        # Model directly says no hardhat
        Detection("NO-Hardhat", 0.88, BBox(110, 45, 240, 140)),
        # Model directly says no vest
        Detection("NO-Safety Vest", 0.82, BBox(105, 130, 245, 280)),
        # Another worker who fell
        Detection("Fall-Detected", 0.95, BBox(300, 350, 500, 480)),
        # A ladder with no cone nearby
        Detection("Ladder", 0.76, BBox(600, 100, 680, 500)),
    ]

    events = engine.evaluate(
        detections=sample_detections,
        frame_id=42,
        timestamp="2025-08-01T09:15:00Z",
        screenshot_path="/frames/frame_042.jpg",
    )

    print(f"Violations detected: {len(events)}\n")
    for ev in events:
        print(json.dumps(ev.to_dict(), indent=2))