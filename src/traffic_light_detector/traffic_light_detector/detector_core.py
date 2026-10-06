"""ROS-independent OpenCV detector for a horizontal four-aspect signal.

The detector deliberately does not infer RED or YELLOW from a white blob alone.
It generates a four-slot housing hypothesis around every plausible illuminated
blob, validates the dark lens rims and low-saturation housing, measures a
coloured halo outside a saturated core, and then applies temporal filtering.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import os

import cv2
import numpy as np
import onnxruntime as ort


BBox = Tuple[int, int, int, int]
STATES = ("RED", "YELLOW", "LEFT", "GREEN")
SLOT_STATE = {0: "RED", 1: "YELLOW", 2: "LEFT", 3: "GREEN"}

_YOLO_MODEL_PATH = "/home/user/ruby_ws/traffic_light_yolo/yolo26n_traffic_light.onnx"
_YOLO_IN_SIZE = 320
_YOLO_CONF_TH = 0.2


def _yolo_letterbox(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


DEFAULT_CONFIG: Dict[str, object] = {
    "search_roi.x": 0.15, "search_roi.y": 0.20,
    "search_roi.w": 0.70, "search_roi.h": 0.35,
    "slot_boundaries": [0.0, 0.25, 0.50, 0.75, 1.0],
    "slot_margin": 0.10,
    "hsv.red1.h_min": 0, "hsv.red1.h_max": 12,
    "hsv.red1.s_min": 55, "hsv.red1.s_max": 255,
    "hsv.red1.v_min": 75, "hsv.red1.v_max": 255,
    "hsv.red2.h_min": 168, "hsv.red2.h_max": 179,
    "hsv.red2.s_min": 55, "hsv.red2.s_max": 255,
    "hsv.red2.v_min": 75, "hsv.red2.v_max": 255,
    "hsv.yellow.h_min": 12, "hsv.yellow.h_max": 38,
    "hsv.yellow.s_min": 45, "hsv.yellow.s_max": 255,
    "hsv.yellow.v_min": 90, "hsv.yellow.v_max": 255,
    "hsv.green.h_min": 38, "hsv.green.h_max": 102,
    "hsv.green.s_min": 35, "hsv.green.s_max": 255,
    "hsv.green.v_min": 70, "hsv.green.v_max": 255,
    "auxiliary.red_normalized_min": 0.39,
    "auxiliary.red_dominance_min": 18.0,
    "auxiliary.lab_a_min": 136.0,
    "auxiliary.yellow_rg_normalized_min": 0.69,
    "auxiliary.yellow_blue_normalized_max": 0.25,
    "auxiliary.yellow_dominance_min": 18.0,
    "auxiliary.lab_b_min": 142.0,
    "auxiliary.green_normalized_min": 0.36,
    "auxiliary.green_dominance_min": 8.0,
    "auxiliary.lab_a_max": 128.0,
    "white.v_min": 238, "white.s_max": 48,
    "white.channel_saturated_min": 250,
    "halo.dilation_size": 9, "halo.min_color_ratio": 0.045,
    "morphology.open_kernel": 3, "morphology.close_kernel": 5,
    "morphology.iterations": 1,
    "contour.min_area": 5.0, "contour.max_area": 4500.0,
    "contour.min_circularity": 0.24,
    "contour.min_aspect": 0.42, "contour.max_aspect": 2.40,
    "bbox.min_width": 90, "bbox.min_height": 24,
    "bbox.max_width_ratio": 0.95,
    "bbox.aspect_min": 2.7, "bbox.aspect_max": 5.8,
    "bbox.pitch_scales": [0.75, 1.00, 1.30, 1.60, 2.00],
    "housing.dark_v_max": 105, "housing.low_s_max": 70,
    "housing.min_dark_ratio": 0.035,
    "housing.min_low_s_ratio": 0.42,
    "housing.min_mid_gray_ratio": 0.38,
    "housing.min_confidence": 0.88,
    "housing.min_structure_score": 0.24,
    "housing.min_circle_slots": 3,
    "housing.min_circle_match": 0.20,
    "candidate.min_confidence": 0.38,
    "candidate.white_light_threshold": 0.68,
    "candidate.acquire_max_center_y_ratio": 0.38,
    "candidate.acquire_min_center_x_ratio": 0.25,
    "candidate.acquire_max_center_x_ratio": 0.66,
    "candidate.max_count": 8,
    "candidate.temporal_weight": 0.20,
    "candidate.structure_weight": 0.34,
    "candidate.housing_weight": 0.18,
    "candidate.halo_weight": 0.16,
    "candidate.shape_weight": 0.12,
    "candidate.classification_weight": 0.28,
    "candidate.classification_margin_weight": 0.02,
    "candidate.nms_iou_threshold": 0.90,
    "score.pixel_ratio_weight": 0.20,
    "score.halo_color_weight": 0.25,
    "score.brightness_weight": 0.10,
    "score.contour_area_weight": 0.15,
    "score.circularity_weight": 0.10,
    "score.slot_position_weight": 0.10,
    "score.housing_structure_weight": 0.10,
    "score.target_pixel_ratio": 0.10,
    "score.min_score": 0.43, "score.margin": 0.055,
    "score.white_penalty_weight": 0.38,
    "score.left_priority_margin": 0.10,
    # Disabled by default so a conventional standalone LEFT lamp remains valid.
    # The supplied course config enables this because its protected-left phase
    # always illuminates LEFT and straight GREEN together.
    "score.left_simultaneous_green_min": 0.0,
    "score.left_pair.left_min_pixel_ratio": 0.0,
    "score.left_pair.left_min_area_ratio": 0.0,
    "score.left_pair.left_min_brightness": 0.0,
    "score.left_pair.green_min_pixel_ratio": 0.0,
    "score.left_pair.green_min_area_ratio": 0.0,
    "score.left_pair.green_min_brightness": 0.0,
    "score.left_pair.left_min_score": 0.43,
    "score.left_pair.left_min_strict_hsv_ratio": 0.0,
    "score.left_pair.left_min_mean_saturation": 0.0,
    "score.left_pair.left_min_halo_ratio": 0.0,
    "score.left_pair.left_min_position": 0.0,
    "score.left_pair.left_max_blob_aspect": 999.0,
    "score.left_pair.left_min_blob_extent": 0.0,
    "score.left_pair.green_min_strict_hsv_ratio": 0.0,
    "score.left_pair.green_min_mean_saturation": 0.0,
    "score.left_pair.green_min_halo_ratio": 0.0,
    "score.left_pair.green_min_position": 0.0,
    "score.left_pair.green_max_blob_aspect": 999.0,
    "score.left_pair.green_min_blob_extent": 0.0,
    "score.left_pair.min_area_similarity": 0.0,
    "score.left_pair.max_vertical_offset_ratio": 999.0,
    "score.left_pair.min_pitch_ratio": 0.0,
    "score.left_pair.max_pitch_ratio": 999.0,
    "score.left_pair.require_fresh_detection": False,
    "score.left_pair.strong_left_brightness": 0.0,
    "score.left_pair.strong_evidence_timeout_sec": 0.0,
    "tracking.enabled": True, "tracking.tracker_type": "CSRT",
    "tracking.redetect_interval": 5, "tracking.max_lost_frames": 5,
    "tracking.search_expand_ratio": 1.8,
    "tracking.bbox_smoothing_alpha": 0.40,
    "tracking.max_center_jump_ratio": 0.40,
    "tracking.max_size_jump_ratio": 0.60,
    "tracking.enforce_position_gate": True,
    "tracking.min_roi_overlap_ratio": 0.75,
    "tracking.reacquire_cooldown_sec": 4.0,
    "temporal_filter.confirm_frames": 3,
    "temporal_filter.left_confirm_frames": 3,
    "temporal_filter.history_size": 5,
    "temporal_filter.unknown_hold_frames": 3,
    "temporal_filter.state_hold_timeout_sec": 0.50,
    "debug.show_rejected_candidates": False,
    "debug.max_rejected_candidates": 4,
    "debug.proposal_min_frames": 3,
}


def default_config() -> Dict[str, object]:
    """Return a mutable copy of all core parameters."""
    return {key: (list(value) if isinstance(value, list) else value)
            for key, value in DEFAULT_CONFIG.items()}


def odd_kernel(value: object) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def sanitized_hsv(config: Mapping[str, object], name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return safe ordered OpenCV HSV endpoints for one configured range."""
    values = []
    limits = ((0, 179), (0, 255), (0, 255))
    for channel, (low_limit, high_limit) in zip(("h", "s", "v"), limits):
        lo = int(config.get(f"hsv.{name}.{channel}_min", low_limit))
        hi = int(config.get(f"hsv.{name}.{channel}_max", high_limit))
        lo, hi = sorted((max(low_limit, min(high_limit, lo)),
                         max(low_limit, min(high_limit, hi))))
        values.append((lo, hi))
    return (np.array([item[0] for item in values], dtype=np.uint8),
            np.array([item[1] for item in values], dtype=np.uint8))


def clip_bbox(bbox: Sequence[float], shape: Sequence[int]) -> Optional[BBox]:
    """Clip x/y/w/h to an image; return None for an empty result."""
    height, width = int(shape[0]), int(shape[1])
    if len(bbox) != 4 or height <= 0 or width <= 0:
        return None
    x, y, w, h = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    x1 = max(0, min(width, int(math.floor(x))))
    y1 = max(0, min(height, int(math.floor(y))))
    x2 = max(0, min(width, int(math.ceil(x + w))))
    y2 = max(0, min(height, int(math.ceil(y + h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def bbox_iou(first: Optional[BBox], second: Optional[BBox]) -> float:
    if first is None or second is None:
        return 0.0
    ax, ay, aw, ah = first; bx, by, bw, bh = second
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 0.0


def slot_boxes(bbox: BBox, boundaries: Sequence[float], margin: float,
               shape: Optional[Sequence[int]] = None) -> List[BBox]:
    """Split a detected housing into four margin-trimmed slots."""
    if len(boundaries) != 5:
        boundaries = (0.0, 0.25, 0.50, 0.75, 1.0)
    values = np.clip(np.sort(np.asarray(boundaries, dtype=float)), 0.0, 1.0)
    x, y, w, h = bbox; result: List[BBox] = []
    margin = max(0.0, min(0.45, float(margin)))
    for left, right in zip(values[:-1], values[1:]):
        sx1 = x + w * left; sx2 = x + w * right
        pad_x = (sx2 - sx1) * margin; pad_y = h * margin
        candidate = (sx1 + pad_x, y + pad_y,
                     max(1.0, sx2 - sx1 - 2 * pad_x), max(1.0, h - 2 * pad_y))
        clipped = clip_bbox(candidate, shape if shape is not None else (y + h, x + w))
        result.append(clipped if clipped is not None else (0, 0, 1, 1))
    return result


def _contours(mask: np.ndarray) -> List[np.ndarray]:
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return found[0] if len(found) == 2 else found[1]


def _circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour); perimeter = cv2.arcLength(contour, True)
    return float(4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0


def _disk_masks(shape: Sequence[int], center: Tuple[float, float],
                inner_radius: float, outer_radius: float) -> Tuple[np.ndarray, np.ndarray]:
    inner = np.zeros(shape[:2], dtype=np.uint8)
    outer = np.zeros(shape[:2], dtype=np.uint8)
    point = (int(round(center[0])), int(round(center[1])))
    cv2.circle(inner, point, max(1, int(round(inner_radius))), 255, -1)
    cv2.circle(outer, point, max(2, int(round(outer_radius))), 255, -1)
    return inner, cv2.subtract(outer, inner)


@dataclass
class Candidate:
    bbox: BBox
    lamp_center: Tuple[float, float]
    assigned_slot: int
    confidence: float
    white_light_score: float
    structure_score: float
    housing_score: float
    halo_ratios: Dict[str, float]
    circularity: float
    temporal_score: float
    rejected: bool = False
    reject_reason: str = ""
    circle_slots: int = 0


@dataclass
class DetectionResult:
    state: str = "UNKNOWN"
    raw_state: str = "UNKNOWN"
    scores: Dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in STATES})
    bbox: Optional[BBox] = None
    detection_confidence: float = 0.0
    white_light_score: float = 1.0
    tracking_state: str = "SEARCHING"
    lost_frames: int = 0
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    candidates: List[Candidate] = field(default_factory=list)
    overlay: Optional[np.ndarray] = None
    processing_fps: float = 0.0
    halo_ratios: Dict[str, float] = field(default_factory=dict)
    proposal_bbox: Optional[BBox] = None
    proposal_state: str = "UNKNOWN"
    left_pair_reason: str = ""
    circle_slots: int = 0


class TemporalStateFilter:
    """Debounce states and put a finite time bound on UNKNOWN holding."""

    def __init__(self, confirm_frames: int = 3, history_size: int = 5,
                 unknown_hold_frames: int = 3, hold_timeout: float = 0.5,
                 left_confirm_frames: Optional[int] = None):
        self.confirm_frames = max(1, int(confirm_frames))
        self.left_confirm_frames = max(
            self.confirm_frames,
            int(left_confirm_frames if left_confirm_frames is not None else confirm_frames))
        self.history = deque(maxlen=max(1, int(history_size), self.left_confirm_frames))
        self.unknown_hold_frames = max(0, int(unknown_hold_frames))
        self.hold_timeout = max(0.0, float(hold_timeout))
        self.state = "UNKNOWN"; self.pending = "UNKNOWN"; self.pending_count = 0
        self.unknown_count = 0; self.last_valid_time: Optional[float] = None

    def update(self, candidate: str, stamp: float) -> str:
        candidate = candidate if candidate in STATES else "UNKNOWN"
        self.history.append(candidate)
        if candidate == "UNKNOWN":
            self.unknown_count += 1; self.pending = "UNKNOWN"; self.pending_count = 0
            expired = (self.last_valid_time is None or
                       stamp - self.last_valid_time > self.hold_timeout)
            if expired or self.unknown_count > self.unknown_hold_frames:
                self.state = "UNKNOWN"
            return self.state
        self.unknown_count = 0
        if candidate == self.state:
            self.pending = candidate; self.pending_count = 0; self.last_valid_time = stamp
            return self.state
        # A distant LEFT arrow can disappear for one or two frames while the
        # simultaneously lit straight-GREEN lamp remains stable.  Confirm LEFT
        # from a rolling window rather than demanding strict consecutiveness.
        # The current frame still has to be a fully validated LEFT pair, so an
        # old observation can never switch the published state by itself.
        if candidate == "LEFT":
            recent_count = sum(value == candidate for value in self.history)
            self.pending = candidate; self.pending_count = recent_count
            if recent_count >= self.left_confirm_frames:
                self.state = candidate; self.pending_count = 0
                self.last_valid_time = stamp
            return self.state
        if candidate != self.pending:
            self.pending = candidate; self.pending_count = 1
        else:
            self.pending_count += 1
        required = self.confirm_frames
        recent_count = sum(value == candidate for value in self.history)
        if self.pending_count >= required and recent_count >= required:
            self.state = candidate; self.pending_count = 0; self.last_valid_time = stamp
        return self.state


class TrafficLightDetectorCore:
    """Stateful OpenCV detector with optional CSRT/KCF tracking fallback."""

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        self.config = default_config(); self.config.update(dict(config or {}))
        self.frame_index = 0; self.previous_bbox: Optional[BBox] = None
        self.tracking_state = "SEARCHING"; self.lost_frames = 0
        self.tracker = None; self.last_detection_confidence = 0.0
        self.last_stamp: Optional[float] = None
        self.reacquire_after_stamp = float("-inf")
        self.left_strong_until_stamp = float("-inf")
        self.had_confirmed_track = False
        self.last_time = time.monotonic(); self.fps_ema = 0.0
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self._yolo_sess = ort.InferenceSession(
            _YOLO_MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
        self._yolo_input_name = self._yolo_sess.get_inputs()[0].name
        self._reset_filter()

    def _reset_filter(self) -> None:
        c = self.config
        self.temporal = TemporalStateFilter(
            c["temporal_filter.confirm_frames"], c["temporal_filter.history_size"],
            c["temporal_filter.unknown_hold_frames"],
            c["temporal_filter.state_hold_timeout_sec"],
            c["temporal_filter.left_confirm_frames"])

    def update_config(self, values: Mapping[str, object], reset_filter: bool = False) -> None:
        self.config.update(dict(values))
        if reset_filter:
            self._reset_filter()

    def reset(self) -> None:
        self.frame_index = 0; self.previous_bbox = None; self.tracker = None
        self.tracking_state = "SEARCHING"; self.lost_frames = 0
        self.last_detection_confidence = 0.0; self._reset_filter()
        self.last_stamp = None; self.reacquire_after_stamp = float("-inf")
        self.left_strong_until_stamp = float("-inf")
        self.had_confirmed_track = False

    def _search_bbox(self, shape: Sequence[int]) -> BBox:
        h, w = shape[:2]; c = self.config
        normalized = (float(c["search_roi.x"]) * w, float(c["search_roi.y"]) * h,
                      float(c["search_roi.w"]) * w, float(c["search_roi.h"]) * h)
        return clip_bbox(normalized, shape) or (0, 0, w, h)

    def build_masks(self, frame: np.ndarray, include_circles: bool = True) -> Dict[str, np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        channels = frame.astype(np.float32)
        b, g, r = cv2.split(channels); total = b + g + r + 1.0
        rn, gn, bn = r / total, g / total, b / total
        masks: Dict[str, np.ndarray] = {"hsv": hsv, "lab": lab}
        for name in ("red1", "red2", "yellow", "green"):
            low, high = sanitized_hsv(self.config, name)
            masks[name] = cv2.inRange(hsv, low, high)
        v = hsv[:, :, 2]; la = lab[:, :, 1].astype(np.float32); lb = lab[:, :, 2].astype(np.float32)
        red_aux = ((rn >= float(self.config["auxiliary.red_normalized_min"])) &
                   ((r - np.maximum(g, b)) >= float(self.config["auxiliary.red_dominance_min"])) &
                   (la >= float(self.config["auxiliary.lab_a_min"])) & (v >= 70))
        yellow_aux = (((rn + gn) >= float(self.config["auxiliary.yellow_rg_normalized_min"])) &
                      (bn <= float(self.config["auxiliary.yellow_blue_normalized_max"])) &
                      ((np.minimum(r, g) - b) >= float(self.config["auxiliary.yellow_dominance_min"])) &
                      (lb >= float(self.config["auxiliary.lab_b_min"])) & (v >= 75))
        green_aux = ((gn >= float(self.config["auxiliary.green_normalized_min"])) &
                     ((g - r) >= float(self.config["auxiliary.green_dominance_min"])) &
                     (la <= float(self.config["auxiliary.lab_a_max"])) & (v >= 65))
        masks["red"] = cv2.bitwise_or(cv2.bitwise_or(masks["red1"], masks["red2"]),
                                      red_aux.astype(np.uint8) * 255)
        masks["yellow"] = cv2.bitwise_or(masks["yellow"], yellow_aux.astype(np.uint8) * 255)
        # Keep the strict HSV mask separately.  The combined green mask below
        # intentionally includes normalized-RGB/Lab evidence for recall, but
        # LEFT pair validation must be able to reject grey/white reflections
        # that are admitted only by those auxiliary colour spaces.
        masks["green_hsv"] = masks["green"].copy()
        masks["green"] = cv2.bitwise_or(masks["green"], green_aux.astype(np.uint8) * 255)
        masks["white"] = cv2.inRange(
            hsv, np.array((0, 0, int(self.config["white.v_min"])), np.uint8),
            np.array((179, int(self.config["white.s_max"]), 255), np.uint8))
        masks["saturated_channel"] = (np.max(frame, axis=2) >= int(
            self.config["white.channel_saturated_min"])).astype(np.uint8) * 255
        combined = masks["red"] | masks["yellow"] | masks["green"] | masks["white"]
        open_k = odd_kernel(self.config["morphology.open_kernel"])
        close_k = odd_kernel(self.config["morphology.close_kernel"])
        iterations = max(1, int(self.config["morphology.iterations"]))
        opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
                                  iterations=iterations)
        masks["combined"] = combined
        masks["morphology"] = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
            iterations=iterations)
        # One frame-wide Hough pass is substantially cheaper than running it for
        # every housing hypothesis.  These circles are only structural evidence;
        # none can become a signal without a coloured/white illuminated seed.
        search = self._search_bbox(frame.shape); sx, sy, sw, sh = search
        circles = None
        if include_circles:
            gray = cv2.cvtColor(frame[sy:sy + sh, sx:sx + sw], cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 1.1)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.25, minDist=8,
                                       param1=85, param2=32, minRadius=3,
                                       maxRadius=max(5, min(45, int(min(frame.shape[:2]) * 0.10))))
        if circles is None:
            masks["circles"] = np.empty((0, 3), dtype=np.float32)
        else:
            values = circles[0].astype(np.float32); values[:, 0] += sx; values[:, 1] += sy
            masks["circles"] = values
        return masks

    @staticmethod
    def _mask_ratio(mask: np.ndarray, selector: np.ndarray) -> float:
        count = int(cv2.countNonZero(selector))
        return float(cv2.countNonZero(cv2.bitwise_and(mask, selector)) / count) if count else 0.0

    def _temporal_score(self, bbox: BBox, shape: Sequence[int]) -> float:
        if self.previous_bbox is None:
            return 0.45
        iou = bbox_iou(bbox, self.previous_bbox)
        x, y, w, h = bbox; px, py, pw, ph = self.previous_bbox
        distance = math.hypot(x + w / 2 - px - pw / 2, y + h / 2 - py - ph / 2)
        diagonal = math.hypot(shape[1], shape[0])
        proximity = max(0.0, 1.0 - distance / max(1.0, diagonal * 0.35))
        size_ratio = min(w * h, pw * ph) / max(1.0, max(w * h, pw * ph))
        return float(0.50 * iou + 0.32 * proximity + 0.18 * size_ratio)

    def _structure(self, masks: Mapping[str, np.ndarray], bbox: BBox
                   ) -> Tuple[float, float, List[float], List[float]]:
        hsv = masks["hsv"]; x, y, w, h = bbox
        crop = hsv[y:y + h, x:x + w]
        if crop.size == 0:
            return 0.0, 0.0, [], []
        dark_ratio = float(np.mean(crop[:, :, 2] <= int(self.config["housing.dark_v_max"])))
        low_s_ratio = float(np.mean(crop[:, :, 1] <= int(self.config["housing.low_s_max"])))
        mid_gray_ratio = float(np.mean((crop[:, :, 2] >= 90) & (crop[:, :, 2] <= 220) &
                                       (crop[:, :, 1] <= int(self.config["housing.low_s_max"]))))
        gray = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 120)
        ring_scores = []
        for slot in slot_boxes(bbox, self.config["slot_boundaries"], 0.0, hsv.shape):
            sx, sy, sw, sh = slot; cx, cy = sx + sw / 2, sy + sh / 2
            inner, ring = _disk_masks(hsv.shape, (cx, cy), min(sw, sh) * 0.24,
                                      min(sw, sh) * 0.50)
            ring_crop = ring[y:y + h, x:x + w]
            ring_pixels = cv2.countNonZero(ring_crop)
            if not ring_pixels:
                continue
            local_v = hsv[y:y + h, x:x + w, 2]
            ring_dark = np.mean(local_v[ring_crop > 0] <= int(self.config["housing.dark_v_max"]))
            ring_edge = cv2.countNonZero(cv2.bitwise_and(edges, ring_crop)) / ring_pixels
            ring_scores.append(float(min(1.0, ring_dark / 0.28) * 0.68 +
                                     min(1.0, ring_edge / 0.18) * 0.32))
        ring_structure = float(np.mean(ring_scores)) if ring_scores else 0.0
        circles = np.asarray(masks.get("circles", np.empty((0, 3))), dtype=float)
        slot_options = []
        slots = slot_boxes(bbox, self.config["slot_boundaries"], 0.0, hsv.shape)
        for slot_index, slot in enumerate(slots):
            sx, sy, sw, sh = slot; cx, cy = sx + sw / 2, sy + sh / 2
            pitch = max(1.0, float(sw)); expected_radius = min(sw, sh) * 0.34
            if circles.size:
                distance = np.hypot((circles[:, 0] - cx) / pitch,
                                    (circles[:, 1] - cy) / max(1.0, sh))
                size_error = np.abs(circles[:, 2] - expected_radius) / max(2.0, expected_radius)
                valid = (distance <= 0.48) & (size_error <= 0.85)
                for circle_index in np.flatnonzero(valid):
                    quality = (max(0.0, 1.0 - float(distance[circle_index]) / 0.48) * 0.72 +
                               max(0.0, 1.0 - float(size_error[circle_index]) / 0.85) * 0.28)
                    slot_options.append((quality, slot_index, int(circle_index)))
        # Greedy one-to-one matching prevents one large circular chair/floor
        # feature from being counted as evidence for several adjacent lamps.
        matched = [0.0] * len(slots); used_slots = set(); used_circles = set()
        for quality, slot_index, circle_index in sorted(slot_options, reverse=True):
            if slot_index in used_slots or circle_index in used_circles:
                continue
            matched[slot_index] = float(quality)
            used_slots.add(slot_index); used_circles.add(circle_index)
        minimum_match = float(self.config["housing.min_circle_match"])
        matched_count = sum(value >= minimum_match for value in matched)
        coverage = matched_count / max(1.0, float(len(slots)))
        circle_structure = (0.70 * float(np.mean(matched)) + 0.30 * coverage
                            if matched else 0.0)
        structure = 0.72 * circle_structure + 0.28 * ring_structure
        housing = float(0.35 * min(1.0, low_s_ratio / max(
            0.01, float(self.config["housing.min_low_s_ratio"]))) +
                        0.30 * min(1.0, dark_ratio / max(
            0.005, float(self.config["housing.min_dark_ratio"]))) +
                        0.35 * min(1.0, mid_gray_ratio / max(
            0.01, float(self.config["housing.min_mid_gray_ratio"]))))
        return structure, housing, ring_scores, matched

    def _candidate_halo(self, masks: Mapping[str, np.ndarray], center: Tuple[float, float],
                        pitch: float) -> Tuple[Dict[str, float], float]:
        inner, ring = _disk_masks(masks["hsv"].shape, center, pitch * 0.20, pitch * 0.48)
        ratios = {name: self._mask_ratio(masks[name], ring)
                  for name in ("red", "yellow", "green")}
        inner_white = self._mask_ratio(masks["white"], inner)
        return ratios, inner_white

    def _yolo_detect_box(self, frame: np.ndarray) -> Optional[BBox]:
        """search_roi 안에서 YOLO26n(ONNX)으로 신호등 하우징 박스 위치만 찾는다.
        클래스 예측은 쓰지 않는다 - 색상/상태 판정은 기존 슬롯 기반 로직이 그대로 함."""
        shape = frame.shape
        rx, ry, rw, rh = self._search_bbox(shape)
        roi = frame[ry:ry + rh, rx:rx + rw]
        if roi.size == 0:
            return None
        sq, scale, pad_x, pad_y = _yolo_letterbox(roi, _YOLO_IN_SIZE)
        x = sq.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        x = x[:, ::-1, :, :].copy()
        out = self._yolo_sess.run(None, {self._yolo_input_name: x})[0]
        dets = out[0]  # (300, 6) = [x1, y1, x2, y2, score, cls] - NMS 내장(end-to-end)
        conf = dets[:, 4]
        mask = conf >= _YOLO_CONF_TH
        if not np.any(mask):
            return None
        best = dets[mask][np.argmax(dets[mask, 4])]
        bx1, by1, bx2, by2 = best[:4]
        bx1 = (bx1 - pad_x) / scale + rx
        by1 = (by1 - pad_y) / scale + ry
        bx2 = (bx2 - pad_x) / scale + rx
        by2 = (by2 - pad_y) / scale + ry
        return clip_bbox((bx1, by1, bx2 - bx1, by2 - by1), shape)

    def _hypotheses(self, frame: np.ndarray, masks: Mapping[str, np.ndarray]) -> List[Candidate]:
        bbox = self._yolo_detect_box(frame)
        if bbox is None:
            return []
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        pitch = max(1.0, w / 4.0)  # YOLO 박스는 이미 4슬롯 하우징 전체이므로 폭/4가 슬롯 피치
        structure, housing, _, circle_matches = self._structure(masks, bbox)
        minimum_circle_match = float(self.config["housing.min_circle_match"])
        circle_slots = sum(v >= minimum_circle_match for v in circle_matches)
        halo, inner_white = self._candidate_halo(masks, (cx, cy), pitch)
        halo_strength = max(halo.values()) if halo else 0.0
        temporal = self._temporal_score(bbox, frame.shape)
        no_halo = max(0.0, 1.0 - halo_strength / max(
            0.005, float(self.config["halo.min_color_ratio"])))
        white_score = float(np.clip(
            0.30 * inner_white + 0.27 * no_halo + 0.28 * (1.0 - structure) +
            0.15 * (1.0 - temporal), 0.0, 1.0))
        weights = [float(self.config[key]) for key in (
            "candidate.temporal_weight", "candidate.structure_weight",
            "candidate.housing_weight", "candidate.halo_weight",
            "candidate.shape_weight")]
        # YOLO가 이미 형태(shape)를 학습해서 박스 자체를 판단했으므로 shape 컴포넌트는
        # 만점(1.0) 처리 - 컨투어 기반의 circularity 대체.
        components = [temporal, structure, housing,
                      min(1.0, halo_strength / max(
                          0.005, float(self.config["halo.min_color_ratio"]))),
                      1.0]
        confidence = float(np.dot(weights, components) / max(1e-6, sum(weights)))
        # 과다노출(글레어)로 램프 중심이 하얗게 뜨면 색상 판정 자체가 신뢰 불가능하다.
        # 컨투어 기반 시절엔 이 문턱값으로 후보 자체를 리젝트했는데, YOLO 버전으로
        # 바꾸면서 빠졌던 걸 복구한다 - 안 넣으면 과다노출된 RED가 색상 오판정(예:
        # 배경 초록 노이즈에 밀려 GREEN으로 잘못 채점)으로 새는 경로가 된다.
        rejected = white_score >= float(self.config["candidate.white_light_threshold"])
        reject_reason = "WHITE_LIGHT_SCORE" if rejected else ""
        # assigned_slot=-1: 특정 램프 시드에서 역산한 박스가 아니라 YOLO가 하우징 전체를
        # 직접 찾은 박스라는 표시. detect()의 "예상 슬롯 대 실제 분류" 검증을 건너뛴다.
        return [Candidate(bbox, (cx, cy), -1, confidence, white_score,
                          structure, housing, halo, 1.0, temporal,
                          rejected=rejected, reject_reason=reject_reason,
                          circle_slots=circle_slots)]

    def _create_tracker(self):
        kind = str(self.config["tracking.tracker_type"]).upper()
        names = [f"Tracker{kind}_create", "TrackerCSRT_create", "TrackerKCF_create"]
        for namespace in (cv2, getattr(cv2, "legacy", None)):
            if namespace is None:
                continue
            for name in names:
                factory = getattr(namespace, name, None)
                if factory is not None:
                    try:
                        return factory()
                    except cv2.error:
                        pass
        return None

    def _init_tracker(self, frame: np.ndarray, bbox: BBox) -> None:
        self.tracker = self._create_tracker() if bool(self.config["tracking.enabled"]) else None
        if self.tracker is not None:
            try:
                initialized = self.tracker.init(frame, tuple(int(v) for v in bbox))
                if initialized is False:
                    self.tracker = None
            except (cv2.error, TypeError):
                self.tracker = None

    def _tracked_bbox(self, frame: np.ndarray) -> Optional[BBox]:
        if self.tracker is None:
            return None
        try:
            ok, value = self.tracker.update(frame)
            bbox = clip_bbox(value, frame.shape) if ok else None
            if bbox is None:
                return None
            if bool(self.config["tracking.enforce_position_gate"]):
                x, y, w, h = bbox
                search = self._search_bbox(frame.shape); rx, ry, rw, rh = search
                # IoU is unsuitable when the ROI is much larger than a signal;
                # convert the clipped intersection to a fraction of the signal.
                ix1, iy1 = max(x, rx), max(y, ry)
                ix2, iy2 = min(x + w, rx + rw), min(y + h, ry + rh)
                overlap = (max(0, ix2 - ix1) * max(0, iy2 - iy1) /
                           max(1.0, float(w * h)))
                cx, cy = x + w * 0.5, y + h * 0.5
                x_allowed = (frame.shape[1] * float(
                    self.config["candidate.acquire_min_center_x_ratio"]) <= cx <=
                    frame.shape[1] * float(
                    self.config["candidate.acquire_max_center_x_ratio"]))
                y_allowed = ry <= cy <= ry + rh
                if (overlap < float(self.config["tracking.min_roi_overlap_ratio"]) or
                        not x_allowed or not y_allowed):
                    return None
            return bbox
        except (cv2.error, TypeError):
            return None

    def _smooth_bbox(self, current: BBox) -> BBox:
        if self.previous_bbox is None:
            return current
        alpha = float(np.clip(self.config["tracking.bbox_smoothing_alpha"], 0.0, 1.0))
        values = [int(round(alpha * new + (1.0 - alpha) * old))
                  for new, old in zip(current, self.previous_bbox)]
        return tuple(values)  # type: ignore[return-value]

    def _slot_feature(self, masks: Mapping[str, np.ndarray], bbox: BBox,
                      slot_index: int, state: str, detection_confidence: float,
                      structure: float, white_light_score: float) -> Tuple[float, Dict[str, float]]:
        slot = slot_boxes(bbox, self.config["slot_boundaries"],
                          float(self.config["slot_margin"]), masks["hsv"].shape)[slot_index]
        x, y, w, h = slot; selector = np.zeros(masks["hsv"].shape[:2], np.uint8)
        selector[y:y + h, x:x + w] = 255
        color_name = "green" if state in ("LEFT", "GREEN") else state.lower()
        color_mask = masks[color_name]
        pixel_ratio = self._mask_ratio(color_mask, selector)
        hsv_crop = masks["hsv"][y:y + h, x:x + w]
        brightness = float(np.percentile(hsv_crop[:, :, 2], 90) / 255.0) if hsv_crop.size else 0.0
        local_mask = cv2.bitwise_and(color_mask, selector)[y:y + h, x:x + w]
        contours = _contours(local_mask)
        largest = max(contours, key=cv2.contourArea) if contours else None
        area_ratio = 0.0; circularity = 0.0; position = 0.0
        blob_area_ratio = 0.0; blob_aspect = 999.0; blob_extent = 0.0
        centroid_x = x + w / 2; centroid_y = y + h / 2
        mean_saturation = 0.0; strict_hsv_ratio = 0.0; blob_brightness = 0.0
        if largest is not None:
            area = cv2.contourArea(largest); area_ratio = min(1.0, area / max(1.0, w * h * 0.24))
            blob_area_ratio = float(area / max(1.0, w * h))
            circularity = min(1.0, _circularity(largest))
            bx, by, bw, bh = cv2.boundingRect(largest)
            blob_aspect = float(max(bw, bh) / max(1, min(bw, bh)))
            blob_extent = float(area / max(1.0, bw * bh))
            m = cv2.moments(largest)
            if m["m00"]:
                cx, cy = x + m["m10"] / m["m00"], y + m["m01"] / m["m00"]
                centroid_x, centroid_y = cx, cy
                distance = math.hypot(cx - (x + w / 2), cy - (y + h / 2))
                position = max(0.0, 1.0 - distance / max(1.0, math.hypot(w, h) * 0.55))
            component = np.zeros((h, w), np.uint8)
            cv2.drawContours(component, [largest], -1, 255, -1)
            component_pixels = max(1, cv2.countNonZero(component))
            mean_saturation = float(cv2.mean(hsv_crop[:, :, 1], mask=component)[0] / 255.0)
            component_values = hsv_crop[:, :, 2][component > 0]
            if component_values.size:
                blob_brightness = float(np.percentile(component_values, 90) / 255.0)
            strict_mask = masks.get("green_hsv", color_mask)[y:y + h, x:x + w]
            strict_hsv_ratio = float(cv2.countNonZero(
                cv2.bitwise_and(strict_mask, component)) / component_pixels)
        cx, cy = x + w / 2, y + h / 2
        inner, ring = _disk_masks(masks["hsv"].shape, (cx, cy), min(w, h) * 0.20,
                                  min(w, h) * 0.50)
        halo_ratio = self._mask_ratio(color_mask, ring)
        target_ratio = max(0.005, float(self.config["score.target_pixel_ratio"]))
        features = {
            "pixel": min(1.0, pixel_ratio / target_ratio),
            "halo": min(1.0, halo_ratio / max(0.005, float(self.config["halo.min_color_ratio"]))),
            "brightness": brightness, "area": area_ratio,
            "circularity": circularity, "position": position,
            "housing": min(1.0, 0.55 * structure + 0.45 * detection_confidence),
            "pixel_ratio": pixel_ratio, "halo_ratio": halo_ratio,
            "blob_area_ratio": blob_area_ratio, "blob_aspect": blob_aspect,
            "blob_extent": blob_extent, "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y), "mean_saturation": mean_saturation,
            "strict_hsv_ratio": strict_hsv_ratio, "blob_brightness": blob_brightness,
        }
        weight_names = ("pixel_ratio", "halo_color", "brightness", "contour_area",
                        "circularity", "slot_position", "housing_structure")
        feature_names = ("pixel", "halo", "brightness", "area", "circularity", "position", "housing")
        weights = [float(self.config[f"score.{name}_weight"]) for name in weight_names]
        score = float(np.dot(weights, [features[name] for name in feature_names]) /
                      max(1e-6, sum(weights)))
        # RED/YELLOW 전용 "과다노출 페널티"를 제거했다: 밝은 RED/YELLOW LED가
        # 카메라에서 과다노출되면(중심부가 하얗게 뜸) halo가 옅어져서 이 페널티에
        # 걸리는데, GREEN/LEFT는 이 페널티가 아예 없어서 실제로 켜진 RED보다
        # 근거 약한 GREEN이 비대칭적으로 이겨버리는 원인이 됐다.
        return float(np.clip(score, 0.0, 1.0)), features

    def _left_pair_reject_reason(self, scores: Mapping[str, float],
                                 details: Mapping[str, Mapping[str, float]],
                                 bbox: BBox) -> str:
        """Return why the simultaneous LEFT+GREEN lamp pair was rejected."""
        left = details["LEFT"]; green = details["GREEN"]
        c = self.config
        checks = (
            (scores["LEFT"] >= float(c["score.left_pair.left_min_score"]), "left_score"),
            (scores["GREEN"] >= float(c["score.left_simultaneous_green_min"]), "green_score"),
            (left["pixel_ratio"] >= float(c["score.left_pair.left_min_pixel_ratio"]), "left_pixel"),
            (left["area"] >= float(c["score.left_pair.left_min_area_ratio"]), "left_area"),
            (left["blob_brightness"] >= float(
                c["score.left_pair.left_min_brightness"]), "left_brightness"),
            (green["pixel_ratio"] >= float(c["score.left_pair.green_min_pixel_ratio"]), "green_pixel"),
            (green["area"] >= float(c["score.left_pair.green_min_area_ratio"]), "green_area"),
            (green["blob_brightness"] >= float(
                c["score.left_pair.green_min_brightness"]), "green_brightness"),
        )
        for passed, reason in checks:
            if not passed:
                return reason

        for name, feature in (("left", left), ("green", green)):
            feature_checks = (
                (feature["strict_hsv_ratio"] >= float(
                    c[f"score.left_pair.{name}_min_strict_hsv_ratio"]), "hsv"),
                (feature["mean_saturation"] >= float(
                    c[f"score.left_pair.{name}_min_mean_saturation"]), "saturation"),
                (feature["halo_ratio"] >= float(
                    c[f"score.left_pair.{name}_min_halo_ratio"]), "halo"),
                (feature["position"] >= float(
                    c[f"score.left_pair.{name}_min_position"]), "position"),
                (feature["blob_aspect"] <= float(
                    c[f"score.left_pair.{name}_max_blob_aspect"]), "aspect"),
                (feature["blob_extent"] >= float(
                    c[f"score.left_pair.{name}_min_blob_extent"]), "extent"),
            )
            for passed, suffix in feature_checks:
                if not passed:
                    return f"{name}_{suffix}"

        left_area = left["blob_area_ratio"]; green_area = green["blob_area_ratio"]
        area_similarity = min(left_area, green_area) / max(1e-6, max(left_area, green_area))
        slot_height = max(1.0, float(bbox[3]))
        pitch = max(1.0, float(bbox[2]) / 4.0)
        vertical_offset = abs(left["centroid_y"] - green["centroid_y"]) / slot_height
        pitch_ratio = (green["centroid_x"] - left["centroid_x"]) / pitch
        if area_similarity < float(c["score.left_pair.min_area_similarity"]):
            return "area_similarity"
        if vertical_offset > float(c["score.left_pair.max_vertical_offset_ratio"]):
            return "vertical_offset"
        if not (float(c["score.left_pair.min_pitch_ratio"]) <= pitch_ratio <=
                float(c["score.left_pair.max_pitch_ratio"])):
            return "pitch"
        return ""

    def _left_pair_valid(self, scores: Mapping[str, float],
                         details: Mapping[str, Mapping[str, float]],
                         bbox: BBox) -> bool:
        """Require two distinct and geometrically aligned green lamps."""
        return not self._left_pair_reject_reason(scores, details, bbox)

    def _left_evidence_allowed(self, details: Mapping[str, Mapping[str, float]],
                               stamp: float) -> Tuple[bool, bool]:
        """Gate a new LEFT transition with one recent strong LEFT observation."""
        threshold = float(self.config["score.left_pair.strong_left_brightness"])
        strong_now = details["LEFT"]["blob_brightness"] >= threshold
        allowed = (strong_now or stamp <= self.left_strong_until_stamp or
                   self.temporal.state == "LEFT")
        return bool(allowed), bool(strong_now)

    def _classify(self, masks: Mapping[str, np.ndarray], candidate: Candidate
                 ) -> Tuple[str, Dict[str, float], Dict[str, Dict[str, float]]]:
        scores: Dict[str, float] = {}; details: Dict[str, Dict[str, float]] = {}
        for index, state in SLOT_STATE.items():
            scores[state], details[state] = self._slot_feature(
                masks, candidate.bbox, index, state, candidate.confidence,
                candidate.structure_score, candidate.white_light_score)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        state, best = ordered[0]; _, second = ordered[1]
        if best < float(self.config["score.min_score"]):
            return "UNKNOWN", scores, details

        # LEFT is safety-critical for this course.  Do not infer it merely from
        # two similar aggregate scores: both slot 2 and slot 3 must contain an
        # independently measurable illuminated green blob.  This rejects a
        # GREEN-only signal when a drifting bbox places a grey lens/ceiling
        # reflection in slot 2.
        left_pair_valid = self._left_pair_valid(scores, details, candidate.bbox)

        # When both lamps really are lit, LEFT has unconditional priority over
        # straight GREEN regardless of their small exposure-dependent score gap.
        if state in ("LEFT", "GREEN") and left_pair_valid:
            return "LEFT", scores, details
        if state == "LEFT":
            if scores["GREEN"] >= float(self.config["score.min_score"]):
                return "GREEN", scores, details
            return "UNKNOWN", scores, details
        if best - second < float(self.config["score.margin"]):
            return "UNKNOWN", scores, details
        return state, scores, details

    def detect(self, frame: np.ndarray, stamp: Optional[float] = None,
               make_overlay: bool = True) -> DetectionResult:
        start = time.perf_counter(); stamp = time.monotonic() if stamp is None else float(stamp)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3:
            state = self.temporal.update("UNKNOWN", stamp)
            return DetectionResult(state=state, tracking_state=self.tracking_state,
                                   lost_frames=self.lost_frames)
        if self.last_stamp is not None and (stamp < self.last_stamp or stamp - self.last_stamp > 1.0):
            self.reset()
        self.last_stamp = stamp
        self.frame_index += 1
        interval = max(1, int(self.config["tracking.redetect_interval"]))
        redetect = self.tracking_state != "TRACKING" or self.frame_index % interval == 0
        tracked = None if redetect else self._tracked_bbox(frame)
        masks = self.build_masks(frame, include_circles=tracked is None)
        candidates: List[Candidate] = []
        selected: Optional[Candidate] = None
        if tracked is not None:
            structure, housing, _, _ = self._structure(masks, tracked)
            temporal = self._temporal_score(tracked, frame.shape)
            confidence = float(np.clip(self.last_detection_confidence * 0.92 +
                                       0.05 * structure + 0.03 * housing, 0.0, 1.0))
            selected = Candidate(tracked, (tracked[0] + tracked[2] / 2,
                                          tracked[1] + tracked[3] / 2), -1,
                                 confidence, 0.0, structure, housing,
                                 {"red": 0.0, "yellow": 0.0, "green": 0.0},
                                 1.0, temporal, circle_slots=-1)
        else:
            cooldown_active = (self.previous_bbox is None and
                               stamp < self.reacquire_after_stamp)
            candidates = [] if cooldown_active else self._hypotheses(frame, masks)
            # A bright seed produces four possible alignments.  Housing score
            # alone often favours the wrong alignment, especially with a white
            # enclosure.  Select only an alignment whose classified state lands
            # in the slot that generated it, then combine appearance and
            # housing confidence.  This also rejects coloured reflections that
            # happen to sit beside an unrelated horizontal edge.
            ranked = []
            classification_weight = max(0.0, float(
                self.config["candidate.classification_weight"]))
            for item in candidates:
                if item.rejected:
                    continue
                classified = self._classify(masks, item)
                raw, item_scores, item_details = classified
                if raw == "LEFT":
                    left_allowed, _ = self._left_evidence_allowed(item_details, stamp)
                    if not left_allowed:
                        # 원래는 컨투어 다중후보 시절 "슬롯2(LEFT) 시드가 좌회전
                        # 근거 부족하면 옆 슬롯(GREEN)으로 재해석"하던 폴백인데,
                        # 지금은 _hypotheses()가 YOLO 박스 하나만 반환해서 이게
                        # 무조건 걸려 RED 등을 GREEN으로 오판정하는 원인이 됐다.
                        # 근거 부족하면 그냥 확실치 않다고 인정한다.
                        raw = "UNKNOWN"
                expected = SLOT_STATE.get(item.assigned_slot)
                # During a protected-left phase either green lamp may be the
                # stronger hypothesis seed.  A tiny distant LEFT arrow can be
                # too small to seed slot 2, while the straight lamp in slot 3
                # still yields the correct four-slot box.  The strict pair
                # validator below has already proved that both lamps exist.
                pair_seed = raw == "LEFT" and item.assigned_slot in (2, 3)
                # assigned_slot == -1: YOLO가 하우징 전체 박스를 직접 찾은 것이라
                # "이 슬롯 시드에서 나왔으니 이 상태여야 한다"는 검증 자체가 성립하지
                # 않는다. 슬롯 기반 분류 결과를 그대로 신뢰한다.
                yolo_box = item.assigned_slot == -1
                if raw == "UNKNOWN" or raw not in item_scores:
                    continue
                if raw != expected and not pair_seed and not yolo_box:
                    continue
                ordered = sorted(item_scores.values(), reverse=True)
                margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
                quality = (item.confidence + classification_weight * item_scores[raw] +
                           float(self.config["candidate.classification_margin_weight"]) *
                           max(0.0, margin))
                # The supplied four-aspect signal illuminates LEFT and GREEN
                # together during the protected-left phase.  Prefer the box
                # that contains both active slots over a shifted hypothesis
                # that treats the right lamp as a green-only signal.
                if (raw == "LEFT" and item_scores["GREEN"] >=
                        float(self.config["score.min_score"])):
                    quality += float(self.config["score.left_priority_margin"])
                ranked.append((quality, item))
            if ranked:
                _, selected = max(ranked, key=lambda value: value[0])
        if selected is not None:
            bbox = self._smooth_bbox(selected.bbox) if self.previous_bbox else selected.bbox
            selected.bbox = bbox
            # Re-evaluate after optional temporal bbox smoothing; even a small
            # shift can move a compact lamp halo across a slot boundary.
            raw_state, scores, details = self._classify(masks, selected)
            left_pair_reason = self._left_pair_reject_reason(scores, details, bbox)
            if raw_state == "LEFT":
                left_allowed, strong_left_now = self._left_evidence_allowed(details, stamp)
                if strong_left_now:
                    self.left_strong_until_stamp = stamp + max(0.0, float(
                        self.config["score.left_pair.strong_evidence_timeout_sec"]))
                if not left_allowed:
                    raw_state = "GREEN"
                    left_pair_reason = "left_strong_evidence"
            # A tracker only propagates an old rectangle; it does not prove that
            # two lamps exist in the current image.  Never let tracker-only LEFT
            # frames advance a new safety-critical LEFT confirmation.  Once LEFT
            # has been confirmed by fresh detections it may be held between the
            # frequent redetection frames.
            tracker_only_left = (
                raw_state == "LEFT" and tracked is not None and
                self.temporal.state not in ("LEFT", "GREEN") and
                bool(self.config["score.left_pair.require_fresh_detection"]))
            final_state = (self.temporal.state if tracker_only_left else
                           self.temporal.update(raw_state, stamp))
            proposal_state = raw_state
            proposal_ready = (final_state in STATES or
                              (self.temporal.pending == raw_state and
                               self.temporal.pending_count >= max(1, int(
                                   self.config["debug.proposal_min_frames"]))))
            proposal_bbox = bbox if proposal_ready else None
            # SEARCHING candidates must produce a temporally confirmed signal
            # before they are allowed to seed a tracker.  This prevents a
            # structurally coincidental chair/floor patch from latching for
            # several redetection cycles.
            if self.previous_bbox is None and final_state == "UNKNOWN":
                bbox = None; self.tracking_state = "SEARCHING"; self.tracker = None
            else:
                if tracked is None:
                    self._init_tracker(frame, bbox)
                self.previous_bbox = bbox; self.last_detection_confidence = selected.confidence
                self.lost_frames = 0; self.tracking_state = "TRACKING"
                if final_state in STATES:
                    self.had_confirmed_track = True
        else:
            self.lost_frames += 1
            if self.previous_bbox is None:
                self.tracking_state = "SEARCHING"
            elif self.lost_frames <= int(self.config["tracking.max_lost_frames"]):
                self.tracking_state = "LOST"
            else:
                self.tracking_state = "SEARCHING"; self.previous_bbox = None; self.tracker = None
                if self.had_confirmed_track:
                    self.reacquire_after_stamp = stamp + max(0.0, float(
                        self.config["tracking.reacquire_cooldown_sec"]))
                    self.had_confirmed_track = False
                    self._reset_filter()
            raw_state = "UNKNOWN"; scores = {name: 0.0 for name in STATES}; details = {}
            left_pair_reason = "no_candidate"
            final_state = self.temporal.update(raw_state, stamp); bbox = None
            proposal_bbox = None; proposal_state = "UNKNOWN"
        elapsed = max(1e-6, time.perf_counter() - start); instant_fps = 1.0 / elapsed
        self.fps_ema = instant_fps if self.fps_ema <= 0 else 0.9 * self.fps_ema + 0.1 * instant_fps
        result = DetectionResult(
            state=final_state, raw_state=raw_state, scores=scores, bbox=bbox,
            detection_confidence=selected.confidence if selected else 0.0,
            white_light_score=selected.white_light_score if selected else 1.0,
            tracking_state=self.tracking_state, lost_frames=self.lost_frames,
            masks=dict(masks), candidates=candidates, processing_fps=self.fps_ema,
            halo_ratios=selected.halo_ratios if selected else {},
            left_pair_reason=left_pair_reason,
            circle_slots=selected.circle_slots if selected else 0)
        result.proposal_bbox = proposal_bbox
        result.proposal_state = proposal_state
        if make_overlay:
            result.overlay = self.draw_debug(frame, result, details)
        return result

    def draw_debug(self, frame: np.ndarray, result: DetectionResult,
                   details: Optional[Mapping[str, Mapping[str, float]]] = None) -> np.ndarray:
        overlay = frame.copy(); sx, sy, sw, sh = self._search_bbox(frame.shape)
        cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (255, 160, 0), 1)
        cv2.putText(overlay, "Search ROI", (sx + 4, sy + 16), 0, 0.45, (255, 160, 0), 1)
        if bool(self.config["debug.show_rejected_candidates"]):
            maximum = max(0, int(self.config["debug.max_rejected_candidates"]))
            rejected = (candidate for candidate in result.candidates if candidate.rejected)
            for candidate in list(rejected)[:maximum]:
                cx, cy = (int(candidate.lamp_center[0]), int(candidate.lamp_center[1]))
                colour = (80, 80, 255)
                cv2.drawMarker(overlay, (cx, cy), colour, cv2.MARKER_TILTED_CROSS, 7, 1)
                cv2.putText(overlay, "REJECT:" + candidate.reject_reason, (cx + 4, cy - 4),
                            0, 0.30, colour, 1)
        if result.bbox is None and result.proposal_bbox is not None:
            x, y, w, h = result.proposal_bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(overlay, "PROPOSAL:" + result.proposal_state, (x, max(12, y - 4)),
                        0, 0.38, (0, 165, 255), 1, cv2.LINE_AA)
        if result.bbox is not None:
            x, y, w, h = result.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
            for index, slot in enumerate(slot_boxes(result.bbox, self.config["slot_boundaries"],
                                                     0.0, frame.shape)):
                qx, qy, qw, qh = slot
                cv2.rectangle(overlay, (qx, qy), (qx + qw, qy + qh), (255, 255, 0), 1)
                cv2.putText(overlay, SLOT_STATE[index], (qx + 2, qy + 13), 0, 0.34,
                            (255, 255, 0), 1)
            for name, ratio in result.halo_ratios.items():
                cv2.putText(overlay, f"halo {name}:{ratio:.3f}", (x, y + h + 14 + 12 * list(
                    result.halo_ratios).index(name)), 0, 0.34, (0, 220, 255), 1)
        lines = [
            f"state={result.state} raw={result.raw_state} {result.tracking_state} lost={result.lost_frames}",
            " ".join(f"{name}:{result.scores.get(name, 0.0):.2f}" for name in STATES),
            f"det={result.detection_confidence:.2f} white={result.white_light_score:.2f} "
            f"circles={result.circle_slots} fps={result.processing_fps:.1f}",
        ]
        if max(result.scores.get("LEFT", 0.0), result.scores.get("GREEN", 0.0)) >= 0.25:
            lines.append("LEFT pair: " + (result.left_pair_reason or "OK"))
        for index, line in enumerate(lines):
            y = 20 + index * 18
            cv2.rectangle(overlay, (0, y - 15), (min(frame.shape[1], 540), y + 3), (0, 0, 0), -1)
            cv2.putText(overlay, line, (4, y), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return overlay
