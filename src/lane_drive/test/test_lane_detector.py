import cv2
from lane_drive.lane_drive_node import SlidingWindowLaneDetector
import numpy as np


class FakeParameter:

    def __init__(self, value):
        self.value = value


class FakeNode:
    VALUES = {
        'white_min_saturation': 70,
        'white_min_value': 175,
        'yellow_hue_min': 15,
        'yellow_hue_max': 40,
        'yellow_min_saturation': 70,
        'yellow_min_value': 80,
        'yellow_side_min_pixels': 80,
        'yellow_side_deadband_px': 20,
        'blur_size': 5,
        'morphology_size': 5,
        'roi_top_ratio': 0.52,
        'roi_bottom_ratio': 1.0,
        'roi_top_half_width_ratio': 0.28,
        'roi_bottom_margin_ratio': 0.02,
        'detection_bottom_ratio': 1.0,
        'camera_center_offset_px': 0,
        'window_count': 9,
        'window_margin_px': 55,
        'window_min_pixels': 30,
        'minimum_lane_pixels': 160,
        'histogram_start_ratio': 0.72,
        'histogram_min_peak': 8,
        'lookahead_y_ratio': 0.72,
        'default_lane_width_ratio': 0.45,
        'minimum_lane_width_ratio': 0.22,
        'maximum_lane_width_ratio': 0.80,
        'lane_width_filter_alpha': 0.2,
    }

    def __init__(self, overrides=None):
        self.values = dict(self.VALUES)
        self.values.update(overrides or {})

    def get_parameter(self, name):
        return FakeParameter(self.values[name])


def test_detects_two_synthetic_lanes():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(image, (160, 479), (250, 250), (255, 255, 255), 15)
    cv2.line(image, (480, 479), (390, 250), (0, 255, 255), 15)

    result = SlidingWindowLaneDetector(FakeNode()).detect(image)

    assert result.valid
    assert result.reason == 'both lanes'
    assert abs(result.lane_center_x - 320.0) < 15.0
    assert result.yellow_side == 1


def test_reports_yellow_divider_on_left():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(image, (160, 479), (250, 250), (0, 255, 255), 15)
    cv2.line(image, (480, 479), (390, 250), (255, 255, 255), 15)

    result = SlidingWindowLaneDetector(FakeNode()).detect(image)

    assert result.valid
    assert result.yellow_side == -1


def test_rejects_empty_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = SlidingWindowLaneDetector(FakeNode()).detect(image)
    assert not result.valid
    assert result.reason == 'no lane'


def test_excludes_pixels_below_detection_bottom():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    detector = SlidingWindowLaneDetector(
        FakeNode({'detection_bottom_ratio': 0.9})
    )

    mask = detector.make_mask(image)

    assert np.any(mask[:432, :])
    assert not np.any(mask[432:, :])
