#!/usr/bin/env python3
"""Interactive HSV tuner for the lane detector without motor output."""

import os
from pathlib import Path
import re
import tempfile
import time
from typing import Dict, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import yaml


HSV_TRACKBARS = (
    ('White S max', 'white_min_saturation', 255),
    ('White V min', 'white_min_value', 255),
    ('Yellow H min', 'yellow_hue_min', 179),
    ('Yellow H max', 'yellow_hue_max', 179),
    ('Yellow S min', 'yellow_min_saturation', 255),
    ('Yellow V min', 'yellow_min_value', 255),
)


def load_lane_parameters(path: Path) -> Dict[str, object]:
    """Load the lane_drive ROS parameter mapping from a YAML file."""
    with path.open('r', encoding='utf-8') as stream:
        document = yaml.safe_load(stream)
    try:
        parameters = document['lane_drive']['ros__parameters']
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f'{path} does not contain lane_drive.ros__parameters'
        ) from exc
    if not isinstance(parameters, dict):
        raise ValueError(f'lane_drive.ros__parameters in {path} is not a mapping')
    return parameters


def update_yaml_values(path: Path, values: Dict[str, int]) -> None:
    """Replace selected scalar values while preserving YAML comments/layout."""
    original = path.read_text(encoding='utf-8')
    lines = original.splitlines(keepends=True)
    keys = '|'.join(re.escape(key) for key in values)
    parameter_pattern = re.compile(
        rf'^(\s*)({keys})(\s*:\s*)(.*?)$'
    )
    found = set()
    updated_lines = []

    for line in lines:
        body = line.rstrip('\r\n')
        newline = line[len(body):]
        code, marker, comment = body.partition('#')
        match = parameter_pattern.match(code)
        if match is None:
            updated_lines.append(line)
            continue

        key = match.group(2)
        old_value_field = match.group(4)
        trailing_space = old_value_field[len(old_value_field.rstrip()):]
        suffix = f'#{comment}' if marker else ''
        updated_lines.append(
            f'{match.group(1)}{key}{match.group(3)}{int(values[key])}'
            f'{trailing_space}{suffix}{newline}'
        )
        found.add(key)

    missing = set(values) - found
    if missing:
        raise ValueError(
            f'Could not find HSV parameters in {path}: {sorted(missing)}'
        )

    stat = path.stat()
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=path.parent,
            prefix=f'.{path.name}.',
            suffix='.tmp',
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.writelines(updated_lines)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, stat.st_mode)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


class HsvTunerNode(Node):
    """Show live HSV masks and save selected values to lane_drive.yaml."""

    PREVIEW_WINDOW = 'Lane HSV Tuner'
    MASK_WINDOW = 'Lane HSV Masks: WHITE | YELLOW | COMBINED'
    SAVE_BUTTON = (15, 15, 205, 58)

    def __init__(self):
        super().__init__('hsv_tuner')
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('config_file', '')
        self.declare_parameter('runtime_config_file', '')
        self.declare_parameter('calibration_file', '')

        configured_path = str(self.get_parameter('config_file').value)
        if configured_path:
            self.config_path = Path(configured_path).expanduser().resolve()
        else:
            self.config_path = (
                Path(get_package_share_directory('lane_drive'))
                / 'config'
                / 'lane_drive.yaml'
            )
        self.parameters = load_lane_parameters(self.config_path)

        runtime_path = str(self.get_parameter('runtime_config_file').value)
        self.runtime_config_path = (
            Path(runtime_path).expanduser() if runtime_path else None
        )

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion = None
        self.calibration_size: Optional[Tuple[int, int]] = None
        self.rectification_maps = None
        self.rectification_enabled = False
        self.status_message = 'Adjust the sliders, then click SAVE YAML'
        self.status_until = 0.0
        self._load_fisheye_calibration()
        self._create_gui()

        image_topic = str(self.get_parameter('image_topic').value)
        self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f'HSV tuner ready: image={image_topic}, YAML={self.config_path}. '
            'This node never publishes motor commands.'
        )

    def _create_gui(self) -> None:
        cv2.namedWindow(self.PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.PREVIEW_WINDOW, 800, 720)
        cv2.namedWindow(self.MASK_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.MASK_WINDOW, 960, 300)
        for label, key, maximum in HSV_TRACKBARS:
            initial = int(self.parameters[key])
            cv2.createTrackbar(
                label,
                self.PREVIEW_WINDOW,
                int(np.clip(initial, 0, maximum)),
                maximum,
                lambda _value: None,
            )
        cv2.setMouseCallback(self.PREVIEW_WINDOW, self._mouse_callback)

    def _mouse_callback(self, event, x, y, _flags, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x1, y1, x2, y2 = self.SAVE_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            self._save_current_values()

    def _trackbar_values(self) -> Dict[str, int]:
        return {
            key: cv2.getTrackbarPos(label, self.PREVIEW_WINDOW)
            for label, key, _maximum in HSV_TRACKBARS
        }

    def _save_current_values(self) -> None:
        values = self._trackbar_values()
        try:
            targets = [self.config_path]
            if self.runtime_config_path is not None:
                runtime_path = self.runtime_config_path.resolve()
                if runtime_path != self.config_path and runtime_path.exists():
                    targets.append(runtime_path)
            for target in targets:
                update_yaml_values(target, values)
            self.parameters.update(values)
            self.status_message = 'SAVED - restart lane_drive to apply'
            self.status_until = time.monotonic() + 4.0
            saved_targets = ', '.join(str(path) for path in targets)
            self.get_logger().info(
                f'Saved HSV values {values} to {saved_targets}'
            )
        except (OSError, ValueError) as exc:
            self.status_message = f'SAVE FAILED: {exc}'
            self.status_until = time.monotonic() + 6.0
            self.get_logger().error(self.status_message)

    def _load_fisheye_calibration(self) -> None:
        if not bool(self.parameters.get('enable_fisheye_rectification', True)):
            return

        configured_path = str(self.get_parameter('calibration_file').value)
        lane_configured_path = str(self.parameters.get('calibration_file', ''))
        if configured_path:
            calibration_path = Path(configured_path).expanduser()
        elif lane_configured_path:
            calibration_path = Path(lane_configured_path).expanduser()
        else:
            calibration_path = self.config_path.parent / 'fisheye_camera.yaml'
            if not calibration_path.exists():
                calibration_path = (
                    Path(get_package_share_directory('lane_drive'))
                    / 'config'
                    / 'fisheye_camera.yaml'
                )

        try:
            with calibration_path.open('r', encoding='utf-8') as stream:
                calibration = yaml.safe_load(stream)
            if calibration.get('distortion_model') != 'equidistant':
                raise ValueError('distortion_model must be equidistant')
            self.camera_matrix = np.asarray(
                calibration['camera_matrix']['data'], dtype=np.float64
            ).reshape(3, 3)
            self.distortion = np.asarray(
                calibration['distortion_coefficients']['data'], dtype=np.float64
            ).reshape(4, 1)
            self.calibration_size = (
                int(calibration['image_width']),
                int(calibration['image_height']),
            )
            self.rectification_enabled = True
            self.get_logger().info(
                f'Loaded fisheye calibration: {calibration_path}'
            )
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            self.get_logger().error(
                f'Could not load fisheye calibration {calibration_path}: {exc}. '
                'Using the raw image.'
            )

    def _rectify_image(self, image: np.ndarray) -> np.ndarray:
        if not self.rectification_enabled:
            return image

        height, width = image.shape[:2]
        image_size = (width, height)
        if image_size != self.calibration_size:
            self.get_logger().error(
                f'Camera image is {width}x{height}, but calibration is '
                f'{self.calibration_size[0]}x{self.calibration_size[1]}; '
                'using the raw image',
                throttle_duration_sec=2.0,
            )
            return image

        if self.rectification_maps is None:
            balance = float(self.parameters.get('rectification_balance', 0.3))
            balance = float(np.clip(balance, 0.0, 1.0))
            new_camera_matrix = (
                cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    self.camera_matrix,
                    self.distortion,
                    image_size,
                    np.eye(3),
                    balance=balance,
                )
            )
            self.rectification_maps = cv2.fisheye.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion,
                np.eye(3),
                new_camera_matrix,
                image_size,
                cv2.CV_16SC2,
            )
            self.get_logger().info(
                f'Fisheye rectification map ready: balance={balance:.2f}'
            )

        map1, map2 = self.rectification_maps
        return cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def _roi_polygon(self, height: int, width: int) -> np.ndarray:
        roi_top = int(height * float(self.parameters['roi_top_ratio']))
        roi_top = max(0, min(height - 2, roi_top))
        roi_bottom = int(height * float(self.parameters['roi_bottom_ratio']))
        roi_bottom = max(roi_top + 1, min(height - 1, roi_bottom))
        top_half_width = int(
            width * float(self.parameters['roi_top_half_width_ratio'])
        )
        bottom_margin = int(
            width * float(self.parameters['roi_bottom_margin_ratio'])
        )
        center_x = width // 2 + int(
            self.parameters.get('camera_center_offset_px', 0)
        )
        bottom_left = max(0, bottom_margin)
        bottom_right = min(width - 1, width - 1 - bottom_margin)
        return np.array(
            [[
                (bottom_left, height - 1),
                (bottom_left, roi_bottom),
                (max(0, center_x - top_half_width), roi_top),
                (min(width - 1, center_x + top_half_width), roi_top),
                (bottom_right, roi_bottom),
                (bottom_right, height - 1),
            ]],
            dtype=np.int32,
        )

    def _detection_bottom_y(self, height: int) -> int:
        ratio = float(self.parameters.get('detection_bottom_ratio', 1.0))
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return max(1, min(height, int(height * ratio)))

    def _make_masks(self, image: np.ndarray, values: Dict[str, int]):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(
            hsv,
            np.array([0, 0, values['white_min_value']], dtype=np.uint8),
            np.array(
                [179, values['white_min_saturation'], 255], dtype=np.uint8
            ),
        )
        yellow = cv2.inRange(
            hsv,
            np.array(
                [
                    values['yellow_hue_min'],
                    values['yellow_min_saturation'],
                    values['yellow_min_value'],
                ],
                dtype=np.uint8,
            ),
            np.array(
                [values['yellow_hue_max'], 255, 255], dtype=np.uint8
            ),
        )

        combined = cv2.bitwise_or(white, yellow)
        blur_size = int(self.parameters.get('blur_size', 5))
        if blur_size > 1:
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            combined = cv2.GaussianBlur(
                combined, (blur_size, blur_size), 0
            )
            _, combined = cv2.threshold(
                combined, 127, 255, cv2.THRESH_BINARY
            )

        morphology_size = int(self.parameters.get('morphology_size', 5))
        if morphology_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (morphology_size, morphology_size)
            )
            combined = cv2.morphologyEx(
                combined, cv2.MORPH_CLOSE, kernel
            )

        height, width = combined.shape
        roi_mask = np.zeros_like(combined)
        polygon = self._roi_polygon(height, width)
        cv2.fillPoly(roi_mask, polygon, 255)
        detection_bottom_y = self._detection_bottom_y(height)
        roi_mask[detection_bottom_y:, :] = 0
        return (
            cv2.bitwise_and(white, roi_mask),
            cv2.bitwise_and(yellow, roi_mask),
            cv2.bitwise_and(combined, roi_mask),
            polygon,
            detection_bottom_y,
        )

    @staticmethod
    def _labeled_mask(mask: np.ndarray, label: str) -> np.ndarray:
        panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        return panel

    def _draw_preview(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        polygon: np.ndarray,
        detection_bottom_y: int,
        values: Dict[str, int],
    ) -> np.ndarray:
        preview = image.copy()
        overlay = np.zeros_like(preview)
        overlay[:, :, 1] = mask
        preview = cv2.addWeighted(preview, 1.0, overlay, 0.35, 0.0)
        cv2.polylines(preview, polygon, True, (255, 255, 0), 2)
        if detection_bottom_y < preview.shape[0]:
            cv2.line(
                preview,
                (0, detection_bottom_y),
                (preview.shape[1] - 1, detection_bottom_y),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                preview,
                'DETECTION BOTTOM',
                (10, max(20, detection_bottom_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        x1, y1, x2, y2 = self.SAVE_BUTTON
        cv2.rectangle(preview, (x1, y1), (x2, y2), (40, 180, 40), -1)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            preview,
            'SAVE YAML',
            (x1 + 19, y1 + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )

        if values['yellow_hue_min'] > values['yellow_hue_max']:
            status = 'WARNING: Yellow H min is greater than H max'
            color = (0, 0, 255)
        elif time.monotonic() <= self.status_until:
            status = self.status_message
            color = (0, 255, 255)
        else:
            status = 'Green overlay = final lane mask | S key also saves'
            color = (255, 255, 255)
        cv2.rectangle(
            preview,
            (0, preview.shape[0] - 35),
            (preview.shape[1], preview.shape[0]),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            preview,
            status,
            (10, preview.shape[0] - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
        )
        return preview

    def image_callback(self, msg: Image) -> None:
        try:
            raw_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except CvBridgeError as exc:
            self.get_logger().error(f'Could not convert camera image: {exc}')
            return

        image = self._rectify_image(raw_image)
        values = self._trackbar_values()
        white, yellow, combined, polygon, detection_bottom_y = (
            self._make_masks(image, values)
        )
        preview = self._draw_preview(
            image, combined, polygon, detection_bottom_y, values
        )

        panels = [
            self._labeled_mask(white, 'WHITE'),
            self._labeled_mask(yellow, 'YELLOW'),
            self._labeled_mask(combined, 'COMBINED + ROI'),
        ]
        target_height = max(1, image.shape[0] // 2)
        resized_panels = [
            cv2.resize(
                panel,
                (image.shape[1] // 2, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
            for panel in panels
        ]
        mask_view = np.hstack(resized_panels)

        cv2.imshow(self.PREVIEW_WINDOW, preview)
        cv2.imshow(self.MASK_WINDOW, mask_view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('s'), ord('S')):
            self._save_current_values()
        elif key in (ord('q'), 27):
            rclpy.shutdown()

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except (cv2.error, KeyboardInterrupt):
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HsvTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
