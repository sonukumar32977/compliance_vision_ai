"""
Compliance Vision AI — Video / Image Processor
Handles frame extraction from video files or RTSP streams and
runs YOLOv8 inference on each extracted frame.
"""

from __future__ import annotations
import cv2
import datetime
import numpy as np
from pathlib import Path
from typing import Generator, List, Tuple, Optional, Callable

from src.config import DETECT_CFG, SCREENSHOTS
from src.violation_logic import Detection, BBox, ViolationLogicEngine, parse_yolo_results


# ──────────────────────────────────────────────────────────────
# Model loader (lazy import — requires ultralytics installed)
# ──────────────────────────────────────────────────────────────

def load_yolo_model(model_path: Optional[Path] = None):
    """Load YOLOv8 model. Returns None if model file not found."""
    path = model_path or DETECT_CFG.model_path
    if not path.exists():
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(path))
    except Exception as e:
        print(f"[WARNING] Could not load YOLO model: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# Frame extraction helpers
# ──────────────────────────────────────────────────────────────

def frame_timestamp(frame_idx: int, fps: float) -> str:
    """Convert frame index + fps to HH:MM:SS string."""
    total_seconds = int(frame_idx / max(fps, 1))
    return str(datetime.timedelta(seconds=total_seconds))


def preprocess_frame(frame: np.ndarray, size: int = 640) -> np.ndarray:
    """
    Resize to YOLO input size and apply light denoising for
    low-quality CCTV footage.
    """
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    denoised = cv2.fastNlMeansDenoisingColored(resized, None, 5, 5, 7, 21)
    return denoised


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    violations: list,
) -> np.ndarray:
    """
    Annotate a frame with bounding boxes and labels.
    Violation bboxes are drawn in red; other detections in green.
    """
    annotated = frame.copy()
    violated_ids = set()
    for v in violations:
        for det in v.detections:
            bb = det["bbox"]
            violated_ids.add(tuple(bb))

    for det in detections:
        bb  = det.bbox
        key = (bb.x1, bb.y1, bb.x2, bb.y2)
        color = (0, 0, 255) if key in violated_ids else (0, 200, 50)
        cv2.rectangle(annotated, (int(bb.x1), int(bb.y1)), (int(bb.x2), int(bb.y2)), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (int(bb.x1), int(bb.y1) - th - 4),
                      (int(bb.x1) + tw, int(bb.y1)), color, -1)
        cv2.putText(annotated, label, (int(bb.x1), int(bb.y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def save_screenshot(frame: np.ndarray, frame_id: int, prefix: str = "frame") -> str:
    """Save an annotated frame to the screenshots folder. Returns file path."""
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    fname = SCREENSHOTS / f"{prefix}_{frame_id:06d}.jpg"
    cv2.imwrite(str(fname), frame)
    return str(fname)


# ──────────────────────────────────────────────────────────────
# Main processor class
# ──────────────────────────────────────────────────────────────

class VideoProcessor:
    """
    Processes a video file or image, runs YOLO, and emits
    (frame, detections, violations, metadata) tuples via a generator.

    Parameters
    ----------
    source          : Path to video file, image file, or RTSP URL string
    yolo_model      : Loaded YOLO model (or None for demo/debug mode)
    logic_engine    : ViolationLogicEngine instance
    frame_rate      : Frames to sample per second (0 = every frame)
    camera_id       : Camera identifier string
    progress_cb     : Optional callback(current_frame, total_frames) for UI
    """

    def __init__(
        self,
        source,
        yolo_model=None,
        logic_engine: Optional[ViolationLogicEngine] = None,
        frame_rate: int = DETECT_CFG.frame_rate,
        camera_id: str = "CAM-01",
        progress_cb: Optional[Callable] = None,
    ):
        self.source       = str(source)
        self.model        = yolo_model
        self.engine       = logic_engine or ViolationLogicEngine(camera_id=camera_id)
        self.frame_rate   = frame_rate
        self.camera_id    = camera_id
        self.progress_cb  = progress_cb

    # ------------------------------------------------------------------
    def process_image(self, image_path: Path):
        """Process a single image file. Returns (detections, violations, screenshot_path)."""
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")

        detections, violations, screenshot_path = self._infer_frame(
            frame, frame_id=0, timestamp="00:00:00"
        )
        return detections, violations, screenshot_path

    # ------------------------------------------------------------------
    def process_video(self) -> Generator:
        """
        Generator that yields one result per sampled frame:
            (frame_id, timestamp, detections, violations, screenshot_path)
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")

        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_step = max(1, int(fps / self.frame_rate)) if self.frame_rate > 0 else 1

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_step == 0:
                    ts = frame_timestamp(frame_idx, fps)
                    detections, violations, screenshot_path = self._infer_frame(
                        frame, frame_id=frame_idx, timestamp=ts
                    )
                    if self.progress_cb:
                        self.progress_cb(frame_idx, total)
                    yield frame_idx, ts, detections, violations, screenshot_path

                frame_idx += 1
        finally:
            cap.release()

    # ------------------------------------------------------------------
    def _infer_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: str,
    ) -> Tuple[List[Detection], list, str]:
        """
        Run YOLO inference + violation logic on a single frame.
        Returns (detections, violations, screenshot_path).
        """
        if self.model is not None:
            results = self.model.predict(
                frame,
                conf      = DETECT_CFG.confidence,
                iou       = DETECT_CFG.iou_threshold,
                imgsz     = DETECT_CFG.input_size,
                verbose   = False,
            )
            detections = parse_yolo_results(results[0], DETECT_CFG.classes)
        else:
            # Demo mode — no model loaded
            detections = []

        violations = self.engine.evaluate(
            detections,
            frame_id        = frame_id,
            timestamp       = timestamp,
            screenshot_path = "",   # filled after screenshot saved
        )

        annotated = draw_detections(frame, detections, violations)
        screenshot_path = save_screenshot(annotated, frame_id)

        # Back-fill screenshot path into violations
        for v in violations:
            v.screenshot_path = screenshot_path

        return detections, violations, screenshot_path
