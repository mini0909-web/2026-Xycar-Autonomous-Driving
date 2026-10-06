"""Interactive HighGUI tuner that uses the exact runtime detector core."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import yaml

from .detector_core import DEFAULT_CONFIG, TrafficLightDetectorCore, default_config, slot_boxes
from .traffic_light_detector_node import EXTRA_DEFAULTS, camera_qos, declare_parameters


COLOURS = ("red1", "red2", "yellow", "green", "white", "halo")
CONTROL_WINDOW = "Controls"


def config_document(values: Dict[str, object]) -> Dict[str, object]:
    serializable = {}
    for key, value in values.items():
        if key in DEFAULT_CONFIG or key in EXTRA_DEFAULTS:
            serializable[key] = value.item() if isinstance(value, np.generic) else value
    return {"traffic_light_detector": {"ros__parameters": serializable}}


def save_config(path: str, values: Dict[str, object]) -> None:
    target = Path(os.path.expanduser(path)).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config_document(values), stream, sort_keys=True, allow_unicode=True)


def load_config(path: str) -> Dict[str, object]:
    target = Path(os.path.expanduser(path)).resolve()
    with target.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    for node_name in ("traffic_light_detector", "hsv_tuner"):
        section = document.get(node_name, {})
        if isinstance(section, dict) and isinstance(section.get("ros__parameters"), dict):
            return dict(section["ros__parameters"])
    raise ValueError("YAML has no traffic_light_detector.ros__parameters section")


def region_statistics(frame: np.ndarray, first: Tuple[int, int], second: Tuple[int, int]) -> str:
    x1, x2 = sorted((first[0], second[0])); y1, y2 = sorted((first[1], second[1]))
    x1 = max(0, min(frame.shape[1] - 1, x1)); x2 = max(x1 + 1, min(frame.shape[1], x2))
    y1 = max(0, min(frame.shape[0] - 1, y1)); y2 = max(y1 + 1, min(frame.shape[0], y2))
    crop = frame[y1:y2, x1:x2]; hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    chunks = [f"ROI ({x1},{y1})-({x2},{y2}) n={crop.shape[0] * crop.shape[1]}"]
    for name, data in (("BGR", crop), ("HSV", hsv), ("Lab", lab)):
        flat = data.reshape(-1, 3).astype(float)
        chunks.append(f"{name} mean={np.mean(flat, axis=0).round(1).tolist()} "
                      f"median={np.median(flat, axis=0).round(1).tolist()} "
                      f"p05/p95={np.percentile(flat, [5, 95], axis=0).round(1).tolist()}")
    return " | ".join(chunks)


class HsvTunerNode(Node):
    def __init__(self) -> None:
        super().__init__("hsv_tuner")
        self.values = declare_parameters(self); self.values["gui.enabled"] = bool(
            self.get_parameter("gui.enabled").value)
        self.defaults = dict(self.values); self.core = TrafficLightDetectorCore(self.values)
        self.bridge = CvBridge(); self.gui_ready = False; self.paused = False; self.step_once = False
        self.last_frame: Optional[np.ndarray] = None; self.last_result = None
        self.drag_start: Optional[Tuple[int, int]] = None; self.drag_end: Optional[Tuple[int, int]] = None
        self.sample_text = "click a pixel or drag a region"; self.selected_index = 0
        self.last_gui_frame_wall = float("-inf")
        if not bool(self.values["gui.enabled"]):
            self.get_logger().warning("gui.enabled=false; tuner will subscribe but not open windows")
        else:
            self._init_gui()
        topic = str(self.values["image_topic"])
        message_type = CompressedImage if bool(self.values["input_compressed"]) else Image
        self.subscription = self.create_subscription(message_type, topic, self._image_callback,
                                                     camera_qos(self.values))
        # HighGUI needs its event queue pumped even while rosbag playback is
        # paused.  Keeping waitKey inside the image callback makes every window
        # appear hung whenever no camera frame arrives.
        self.gui_timer = self.create_timer(0.03, self._pump_gui_events)
        self.get_logger().info(f"HSV tuner subscribed to {topic}; s/l/r/p, space/n, 1-4, q/ESC")

    @staticmethod
    def _noop(_value: int) -> None:
        return

    def _init_gui(self) -> None:
        if not os.environ.get("DISPLAY") and os.name != "nt":
            self.get_logger().error("DISPLAY is not set; run detector headless or provide an X server")
            return
        try:
            # Keep the tuner lightweight: controls, an ROI preview, and the
            # currently selected HSV mask are sufficient for calibration.
            for name in (CONTROL_WINDOW, "Original", "HSV Mask"):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Selected", CONTROL_WINDOW, 0, len(COLOURS) - 1, self._noop)
            for channel, maximum in (("H min", 179), ("H max", 179),
                                     ("S min", 255), ("S max", 255),
                                     ("V min", 255), ("V max", 255)):
                cv2.createTrackbar(channel, CONTROL_WINDOW, 0, maximum, self._noop)
            controls = {
                "Open": 31, "Close": 31, "Iterations": 5,
                "Min area": 1000, "Max area": 10000, "Circularity x100": 100,
                "Halo dilation": 41, "Halo ratio x1000": 500,
                "White V": 255, "White S max": 255, "White score x100": 100,
                "Detection conf x100": 100, "Signal score x100": 100,
                "Score margin x1000": 500, "Slot margin x100": 45,
                "ROI x x100": 100, "ROI y x100": 100,
                "ROI w x100": 100, "ROI h x100": 100,
                "BBox min width": 320, "BBox min height": 120,
                "Redetect": 30, "Confirm": 15,
            }
            for name, maximum in controls.items():
                cv2.createTrackbar(name, CONTROL_WINDOW, 0, maximum, self._noop)
            cv2.setMouseCallback("Original", self._mouse)
            self.gui_ready = True; self._write_controls(force_colour=True)
        except cv2.error as error:
            self.get_logger().error(f"HighGUI initialization failed: {error}"); self.gui_ready = False

    def _hsv_values(self, name: str):
        if name in ("red1", "red2", "yellow", "green"):
            return [int(self.values[f"hsv.{name}.{channel}_{bound}"])
                    for channel, bound in (("h", "min"), ("h", "max"),
                                           ("s", "min"), ("s", "max"),
                                           ("v", "min"), ("v", "max"))]
        if name == "white":
            return [0, 179, 0, int(self.values["white.s_max"]),
                    int(self.values["white.v_min"]), 255]
        return [0, 179, 0, 255, 0, 255]

    def _write_controls(self, force_colour: bool = False) -> None:
        if not self.gui_ready:
            return
        index = cv2.getTrackbarPos("Selected", CONTROL_WINDOW)
        if force_colour or index != self.selected_index:
            self.selected_index = index
            for label, value in zip(("H min", "H max", "S min", "S max", "V min", "V max"),
                                    self._hsv_values(COLOURS[index])):
                cv2.setTrackbarPos(label, CONTROL_WINDOW, value)
        mapping = {
            "Open": int(self.values["morphology.open_kernel"]),
            "Close": int(self.values["morphology.close_kernel"]),
            "Iterations": int(self.values["morphology.iterations"]),
            "Min area": int(self.values["contour.min_area"]),
            "Max area": int(self.values["contour.max_area"]),
            "Circularity x100": int(float(self.values["contour.min_circularity"]) * 100),
            "Halo dilation": int(self.values["halo.dilation_size"]),
            "Halo ratio x1000": int(float(self.values["halo.min_color_ratio"]) * 1000),
            "White V": int(self.values["white.v_min"]), "White S max": int(self.values["white.s_max"]),
            "White score x100": int(float(self.values["candidate.white_light_threshold"]) * 100),
            "Detection conf x100": int(float(self.values["candidate.min_confidence"]) * 100),
            "Signal score x100": int(float(self.values["score.min_score"]) * 100),
            "Score margin x1000": int(float(self.values["score.margin"]) * 1000),
            "Slot margin x100": int(float(self.values["slot_margin"]) * 100),
            "ROI x x100": int(float(self.values["search_roi.x"]) * 100),
            "ROI y x100": int(float(self.values["search_roi.y"]) * 100),
            "ROI w x100": int(float(self.values["search_roi.w"]) * 100),
            "ROI h x100": int(float(self.values["search_roi.h"]) * 100),
            "BBox min width": int(self.values["bbox.min_width"]),
            "BBox min height": int(self.values["bbox.min_height"]),
            "Redetect": int(self.values["tracking.redetect_interval"]),
            "Confirm": int(self.values["temporal_filter.confirm_frames"]),
        }
        for label, value in mapping.items():
            cv2.setTrackbarPos(label, CONTROL_WINDOW, max(0, value))

    def _read_controls(self) -> None:
        index = cv2.getTrackbarPos("Selected", CONTROL_WINDOW)
        if index != self.selected_index:
            self._write_controls(force_colour=True); return
        name = COLOURS[index]
        hsv_values = [cv2.getTrackbarPos(label, CONTROL_WINDOW)
                      for label in ("H min", "H max", "S min", "S max", "V min", "V max")]
        if name in ("red1", "red2", "yellow", "green"):
            for (channel, bound), value in zip((("h", "min"), ("h", "max"),
                                                ("s", "min"), ("s", "max"),
                                                ("v", "min"), ("v", "max")), hsv_values):
                self.values[f"hsv.{name}.{channel}_{bound}"] = value
        elif name == "white":
            self.values["white.s_max"] = hsv_values[3]; self.values["white.v_min"] = hsv_values[4]
        getters = {
            "morphology.open_kernel": ("Open", 1.0), "morphology.close_kernel": ("Close", 1.0),
            "morphology.iterations": ("Iterations", 1.0), "contour.min_area": ("Min area", 1.0),
            "contour.max_area": ("Max area", 1.0), "contour.min_circularity": ("Circularity x100", .01),
            "halo.dilation_size": ("Halo dilation", 1.0), "halo.min_color_ratio": ("Halo ratio x1000", .001),
            "white.v_min": ("White V", 1.0), "white.s_max": ("White S max", 1.0),
            "candidate.white_light_threshold": ("White score x100", .01),
            "candidate.min_confidence": ("Detection conf x100", .01),
            "score.min_score": ("Signal score x100", .01), "score.margin": ("Score margin x1000", .001),
            "slot_margin": ("Slot margin x100", .01), "search_roi.x": ("ROI x x100", .01),
            "search_roi.y": ("ROI y x100", .01), "search_roi.w": ("ROI w x100", .01),
            "search_roi.h": ("ROI h x100", .01), "bbox.min_width": ("BBox min width", 1.0),
            "bbox.min_height": ("BBox min height", 1.0), "tracking.redetect_interval": ("Redetect", 1.0),
            "temporal_filter.confirm_frames": ("Confirm", 1.0),
        }
        for key, (label, scale) in getters.items():
            value = cv2.getTrackbarPos(label, CONTROL_WINDOW) * scale
            self.values[key] = int(value) if isinstance(DEFAULT_CONFIG.get(key), int) else value
        self.core.update_config(self.values)

    def _mouse(self, event, x, y, _flags, _userdata) -> None:
        if self.last_frame is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y); self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            self.drag_end = (x, y)
            if abs(x - self.drag_start[0]) > 3 or abs(y - self.drag_start[1]) > 3:
                self.sample_text = region_statistics(self.last_frame, self.drag_start, self.drag_end)
            else:
                self.sample_text = self._pixel_text(x, y)
            self.get_logger().info(self.sample_text); self.drag_start = None

    def _pixel_text(self, x: int, y: int) -> str:
        frame = self.last_frame; x = max(0, min(frame.shape[1] - 1, x)); y = max(0, min(frame.shape[0] - 1, y))
        bgr = frame[y, x]; hsv = cv2.cvtColor(frame[y:y + 1, x:x + 1], cv2.COLOR_BGR2HSV)[0, 0]
        lab = cv2.cvtColor(frame[y:y + 1, x:x + 1], cv2.COLOR_BGR2LAB)[0, 0]
        total = float(np.sum(bgr)) + 1.0; normalized = [round(float(bgr[2] / total), 3),
                                                       round(float(bgr[1] / total), 3),
                                                       round(float(bgr[0] / total), 3)]
        search = self.core._search_bbox(frame.shape); sx, sy, sw, sh = search
        in_search = sx <= x < sx + sw and sy <= y < sy + sh
        bbox = self.last_result.bbox if self.last_result is not None else None; slot = "none"; in_bbox = False
        if bbox is not None:
            bx, by, bw, bh = bbox; in_bbox = bx <= x < bx + bw and by <= y < by + bh
            if in_bbox:
                relative = (x - bx) / max(1.0, bw); slot = str(min(3, max(0, int(relative * 4))))
        return (f"({x},{y}) BGR={bgr.tolist()} HSV={hsv.tolist()} Lab={lab.tolist()} "
                f"normRGB={normalized} slot={slot} search={in_search} bbox={in_bbox}")

    def _selected_mask(self, result):
        name = COLOURS[self.selected_index]
        if name in ("red1", "red2", "yellow", "green", "white"):
            return result.masks[name]
        return result.masks["red"] | result.masks["yellow"] | result.masks["green"]

    def _show(self, frame, result) -> None:
        original = frame.copy(); sx, sy, sw, sh = self.core._search_bbox(frame.shape)
        cv2.rectangle(original, (sx, sy), (sx + sw, sy + sh), (255, 150, 0), 1)
        if result.bbox is not None:
            x, y, w, h = result.bbox; cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 255), 2)
            for box in slot_boxes(result.bbox, self.values["slot_boundaries"], 0.0, frame.shape):
                qx, qy, qw, qh = box; cv2.rectangle(original, (qx, qy), (qx + qw, qy + qh), (255, 255, 0), 1)
        cv2.putText(original, self.sample_text[:100], (5, frame.shape[0] - 8), 0, .38, (0, 255, 255), 1)
        if self.drag_start is not None and self.drag_end is not None:
            cv2.rectangle(original, self.drag_start, self.drag_end, (255, 0, 255), 1)
        selected = self._selected_mask(result)
        cv2.imshow("Original", original)
        cv2.imshow("HSV Mask", selected)

    def _handle_key(self, key: int) -> bool:
        if key in (27, ord("q")):
            rclpy.shutdown(); return False
        if ord("1") <= key <= ord("4"):
            cv2.setTrackbarPos("Selected", CONTROL_WINDOW, key - ord("1")); self._write_controls(True)
        elif key == ord("s"):
            try:
                save_config(str(self.values["gui.config_save_path"]), self.values)
                self.get_logger().info(f"saved {self.values['gui.config_save_path']}")
            except (OSError, ValueError, yaml.YAMLError) as error:
                self.get_logger().error(f"YAML save failed: {error}")
        elif key == ord("l"):
            try:
                self.values.update(load_config(str(self.values["gui.config_save_path"])))
                self.core.update_config(self.values, reset_filter=True); self._write_controls(True)
                self.get_logger().info("configuration reloaded")
            except (OSError, ValueError, yaml.YAMLError) as error:
                self.get_logger().error(f"YAML load failed: {error}")
        elif key == ord("r"):
            self.values.update(default_config()); self.core.update_config(self.values, True); self._write_controls(True)
            self.get_logger().info("core thresholds reset to defaults")
        elif key == ord("p"):
            self.get_logger().info(yaml.safe_dump(config_document(self.values), sort_keys=True))
        elif key == ord(" "):
            self.paused = not self.paused
        elif key == ord("n"):
            self.step_once = True
        return True

    def _image_callback(self, message) -> None:
        now = time.monotonic()
        maximum_fps = max(1.0, float(self.values.get("gui.max_processing_fps", 10.0)))
        if now - self.last_gui_frame_wall < 1.0 / maximum_fps:
            return
        self.last_gui_frame_wall = now
        try:
            frame = (self.bridge.compressed_imgmsg_to_cv2(message, "bgr8")
                     if isinstance(message, CompressedImage)
                     else self.bridge.imgmsg_to_cv2(message, "bgr8"))
        except (CvBridgeError, cv2.error, ValueError) as error:
            self.get_logger().warn(f"image conversion failed: {error}", throttle_duration_sec=2.0); return
        if frame is None or frame.size == 0:
            return
        if not self.gui_ready:
            return
        try:
            self._read_controls()
            if not self.paused or self.step_once or self.last_result is None:
                self.last_frame = frame.copy(); stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
                self.last_result = self.core.detect(
                    self.last_frame, stamp, make_overlay=False)
                self.step_once = False
            self._show(self.last_frame, self.last_result)
        except cv2.error as error:
            self.get_logger().error(f"GUI disabled after HighGUI error: {error}"); self.gui_ready = False

    def _pump_gui_events(self) -> None:
        if not self.gui_ready:
            return
        try:
            self._handle_key(cv2.waitKey(1) & 0xFF)
        except cv2.error as error:
            self.get_logger().error(f"GUI event loop failed: {error}")
            self.gui_ready = False

    def destroy_node(self):
        if self.gui_ready:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args); node = HsvTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
