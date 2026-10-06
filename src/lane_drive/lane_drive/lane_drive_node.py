#!/usr/bin/env python3
"""Detect road lanes from a camera image and publish Xycar motor commands."""

from dataclasses import dataclass
from pathlib import Path
import signal
import time
from typing import Optional

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge, CvBridgeError
from drive_control_msgs.srv import DriveEnable
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int8
import yaml


@dataclass
class LaneResult:
    """Result of one lane detection pass."""

    valid: bool
    lane_center_x: Optional[float]
    left_fit: Optional[np.ndarray]
    right_fit: Optional[np.ndarray]
    mask: np.ndarray
    lookahead_y: int
    confidence: float
    reason: str
    # -1: yellow divider is on the left, +1: on the right, 0: unknown.
    yellow_side: int = 0
    yellow_confidence: float = 0.0


def lane_ready_for_avoidance(
    result: LaneResult,
    image_width: int,
    camera_center_offset_px: float,
    maximum_error_px: float,
) -> bool:
    """Require a yellow divider, at least one lane, and a near-centered vehicle."""
    lane_visible = result.left_fit is not None or result.right_fit is not None
    if not result.valid or not lane_visible or result.yellow_side not in (-1, 1):
        return False
    image_center = image_width * 0.5 + float(camera_center_offset_px)
    return abs(float(result.lane_center_x) - image_center) <= abs(float(maximum_error_px))


def lane_ready_for_handoff(
    result: LaneResult,
    image_width: int,
    camera_center_offset_px: float,
    maximum_error_px: float,
) -> bool:
    """Require both lane boundaries and a centered vehicle before handoff."""
    both_lanes = result.left_fit is not None and result.right_fit is not None
    if not result.valid or not both_lanes or result.yellow_side not in (-1, 1):
        return False
    image_center = image_width * 0.5 + float(camera_center_offset_px)
    return abs(float(result.lane_center_x) - image_center) <= abs(float(maximum_error_px))


def limit_speed_increase(
    target_speed: float,
    previous_speed: float,
    elapsed_sec: float,
    acceleration_limit: float,
    start_speed: float,
) -> float:
    """Limit only upward speed changes, leaving deceleration immediate."""
    target_speed = max(0.0, float(target_speed))
    previous_speed = max(0.0, float(previous_speed))
    acceleration_limit = float(acceleration_limit)

    if acceleration_limit <= 0.0 or target_speed <= previous_speed:
        return target_speed
    if previous_speed <= 0.0:
        return min(target_speed, max(0.0, float(start_speed)))

    elapsed_sec = max(0.0, float(elapsed_sec))
    return min(target_speed, previous_speed + acceleration_limit * elapsed_sec)


class SlidingWindowLaneDetector:
    """Color-mask and sliding-window lane detector."""

    def __init__(self, node: Node):
        self.node = node
        self.estimated_lane_width_px: Optional[float] = None
        self.last_yellow_mask: Optional[np.ndarray] = None

    def _p(self, name):
        return self.node.get_parameter(name).value

    def roi_polygon(self, height: int, width: int) -> np.ndarray:
        """Return the configured trapezoidal ROI as OpenCV polygon points."""
        roi_top = int(height * float(self._p('roi_top_ratio')))
        roi_top = max(0, min(height - 2, roi_top))
        roi_bottom = int(height * float(self._p('roi_bottom_ratio')))
        roi_bottom = max(roi_top + 1, min(height - 1, roi_bottom))
        top_half_width = int(width * float(self._p('roi_top_half_width_ratio')))
        bottom_margin = int(width * float(self._p('roi_bottom_margin_ratio')))
        center_x = width // 2 + int(self._p('camera_center_offset_px'))
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

    def detection_bottom_y(self, height: int) -> int:
        """Return the first image row excluded from lane detection."""
        ratio = float(self._p('detection_bottom_ratio'))
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return max(1, min(height, int(height * ratio)))

    def make_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        white_min_saturation = int(self._p('white_min_saturation'))
        white_min_value = int(self._p('white_min_value'))
        white = cv2.inRange(
            hsv,
            np.array([0, 0, white_min_value], dtype=np.uint8),
            np.array([179, white_min_saturation, 255], dtype=np.uint8),
        )

        yellow_hue_min = int(self._p('yellow_hue_min'))
        yellow_hue_max = int(self._p('yellow_hue_max'))
        yellow_min_saturation = int(self._p('yellow_min_saturation'))
        yellow_min_value = int(self._p('yellow_min_value'))
        yellow = cv2.inRange(
            hsv,
            np.array([yellow_hue_min, yellow_min_saturation, yellow_min_value], dtype=np.uint8),
            np.array([yellow_hue_max, 255, 255], dtype=np.uint8),
        )

        mask = cv2.bitwise_or(white, yellow)
        blur_size = int(self._p('blur_size'))
        if blur_size > 1:
            blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        morphology_size = int(self._p('morphology_size'))
        if morphology_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (morphology_size, morphology_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        height, width = mask.shape
        roi_mask = np.zeros_like(mask)

        # A trapezoid removes much of the vehicle body and roadside clutter while
        # retaining the road immediately in front of the camera.
        polygon = self.roi_polygon(height, width)
        cv2.fillPoly(roi_mask, polygon, 255)
        masked = cv2.bitwise_and(mask, roi_mask)
        self.last_yellow_mask = cv2.bitwise_and(yellow, roi_mask)
        masked[self.detection_bottom_y(height):, :] = 0
        self.last_yellow_mask[self.detection_bottom_y(height):, :] = 0
        return masked

    def _yellow_side(self, height: int, width: int):
        """Estimate which image side contains the yellow lane divider."""
        if self.last_yellow_mask is None:
            return 0, 0.0

        start_y = int(height * float(self._p('histogram_start_ratio')))
        end_y = self.detection_bottom_y(height)
        if end_y <= start_y:
            start_y = max(0, end_y - max(20, height // 6))
        yellow_y, yellow_x = self.last_yellow_mask[start_y:end_y, :].nonzero()
        count = int(yellow_x.size)
        minimum = max(1, int(self._p('yellow_side_min_pixels')))
        if count < minimum:
            return 0, min(1.0, count / float(minimum))

        center_x = width * 0.5 + float(self._p('camera_center_offset_px'))
        median_x = float(np.median(yellow_x))
        deadband = max(0.0, float(self._p('yellow_side_deadband_px')))
        if median_x < center_x - deadband:
            side = -1
        elif median_x > center_x + deadband:
            side = 1
        else:
            side = 0
        confidence = min(1.0, count / float(3 * minimum))
        return side, confidence

    def _fit_lane(self, mask: np.ndarray, base_x: Optional[int]):
        if base_x is None:
            return None, 0

        height, _ = mask.shape
        nonzero_y, nonzero_x = mask.nonzero()
        window_count = max(1, int(self._p('window_count')))
        window_height = max(1, height // window_count)
        margin = int(self._p('window_margin_px'))
        min_pixels = int(self._p('window_min_pixels'))
        current_x = int(base_x)
        lane_indices = []

        for window in range(window_count):
            y_low = height - (window + 1) * window_height
            y_high = height - window * window_height
            x_low = current_x - margin
            x_high = current_x + margin
            indices = np.where(
                (nonzero_y >= y_low)
                & (nonzero_y < y_high)
                & (nonzero_x >= x_low)
                & (nonzero_x < x_high)
            )[0]
            lane_indices.append(indices)
            if len(indices) >= min_pixels:
                current_x = int(np.mean(nonzero_x[indices]))

        if not lane_indices:
            return None, 0
        indices = np.concatenate(lane_indices)
        minimum_lane_pixels = int(self._p('minimum_lane_pixels'))
        if len(indices) < minimum_lane_pixels:
            return None, len(indices)

        try:
            fit = np.polyfit(nonzero_y[indices], nonzero_x[indices], 2)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None, len(indices)
        return fit, len(indices)

    @staticmethod
    def _x_at(fit: Optional[np.ndarray], y: int) -> Optional[float]:
        if fit is None:
            return None
        return float(np.polyval(fit, y))

    def detect(self, image: np.ndarray) -> LaneResult:
        mask = self.make_mask(image)
        height, width = mask.shape
        yellow_side, yellow_confidence = self._yellow_side(height, width)
        histogram_start = int(height * float(self._p('histogram_start_ratio')))
        histogram = np.sum(mask[histogram_start:, :] > 0, axis=0)
        midpoint = width // 2 + int(self._p('camera_center_offset_px'))
        midpoint = max(1, min(width - 1, midpoint))
        histogram_min_peak = int(self._p('histogram_min_peak'))

        left_slice = histogram[:midpoint]
        right_slice = histogram[midpoint:]
        left_base = int(np.argmax(left_slice)) if left_slice.size else None
        right_base = int(np.argmax(right_slice) + midpoint) if right_slice.size else None
        if left_base is not None and histogram[left_base] < histogram_min_peak:
            left_base = None
        if right_base is not None and histogram[right_base] < histogram_min_peak:
            right_base = None

        left_fit, left_count = self._fit_lane(mask, left_base)
        right_fit, right_count = self._fit_lane(mask, right_base)
        lookahead_y = int(height * float(self._p('lookahead_y_ratio')))
        lookahead_y = max(0, min(height - 1, lookahead_y))
        left_x = self._x_at(left_fit, lookahead_y)
        right_x = self._x_at(right_fit, lookahead_y)

        default_width = width * float(self._p('default_lane_width_ratio'))
        lane_width = self.estimated_lane_width_px or default_width
        min_width = width * float(self._p('minimum_lane_width_ratio'))
        max_width = width * float(self._p('maximum_lane_width_ratio'))
        minimum_lane_pixels = int(self._p('minimum_lane_pixels'))
        confidence = min(1.0, (left_count + right_count) / max(1.0, 4.0 * minimum_lane_pixels))

        if left_x is not None and right_x is not None:
            measured_width = right_x - left_x
            if min_width <= measured_width <= max_width:
                alpha = float(self._p('lane_width_filter_alpha'))
                self.estimated_lane_width_px = (
                    measured_width
                    if self.estimated_lane_width_px is None
                    else alpha * measured_width + (1.0 - alpha) * self.estimated_lane_width_px
                )
                lane_center = (left_x + right_x) * 0.5
                confidence = max(confidence, 0.8)
                reason = 'both lanes'
            elif left_count >= right_count:
                right_fit = None
                lane_center = left_x + lane_width * 0.5
                confidence *= 0.55
                reason = 'invalid width; using left lane'
            else:
                left_fit = None
                lane_center = right_x - lane_width * 0.5
                confidence *= 0.55
                reason = 'invalid width; using right lane'
        elif left_x is not None:
            lane_center = left_x + lane_width * 0.5
            confidence *= 0.55
            reason = 'left lane only'
        elif right_x is not None:
            lane_center = right_x - lane_width * 0.5
            confidence *= 0.55
            reason = 'right lane only'
        else:
            return LaneResult(
                False, None, None, None, mask, lookahead_y, 0.0, 'no lane',
                yellow_side, yellow_confidence,
            )

        if not (-0.25 * width <= lane_center <= 1.25 * width):
            return LaneResult(
                False, None, left_fit, right_fit, mask, lookahead_y, 0.0,
                'implausible lane center',
                yellow_side, yellow_confidence,
            )
        return LaneResult(
            True, lane_center, left_fit, right_fit, mask, lookahead_y,
            float(np.clip(confidence, 0.0, 1.0)), reason,
            yellow_side, yellow_confidence,
        )


class LaneDriveNode(Node):
    """ROS 2 node combining perception, steering control, and safe motor output."""

    def __init__(self):
        super().__init__('lane_drive')
        self._declare_parameters()

        self.bridge = CvBridge()
        self.detector = SlidingWindowLaneDetector(self)
        self.rectification_enabled = False
        self.camera_matrix = None
        self.distortion = None
        self.calibration_size = None
        self.rectification_maps = None
        self._load_fisheye_calibration()
        self.enabled = bool(self.get_parameter('auto_start').value)
        self.last_image_time: Optional[float] = None
        self.last_lane_time: Optional[float] = None
        self.filtered_error = 0.0
        self.previous_error = 0.0
        self.last_angle = 0.0
        self.last_commanded_speed = 0.0
        self.last_speed_command_time: Optional[float] = None
        self.stop_published = False

        image_topic = str(self.get_parameter('image_topic').value)
        motor_topic = str(self.get_parameter('motor_topic').value)
        self.motor_pub = self.create_publisher(Float32MultiArray, motor_topic, 10)
        self.error_pub = self.create_publisher(Float32, '/lane/error', 10)
        # Keep the divider-side topic for diagnostics.  ego_lane is deliberately
        # separate: it is a road-lane ID shared with obstacle perception.
        self.yellow_side_pub = self.create_publisher(Int8, '/lane/yellow_side', 10)
        self.ego_lane_pub = self.create_publisher(Int8, '/lane/ego_lane', 10)
        self.avoidance_ready_pub = self.create_publisher(Bool, '/lane/avoidance_ready', 10)
        self.handoff_ready_pub = self.create_publisher(Bool, '/lane/handoff_ready', 10)
        rectified_topic = str(self.get_parameter('rectified_topic').value)
        self.rectified_pub = self.create_publisher(
            Image, rectified_topic, qos_profile_sensor_data
        )
        self.debug_pub = self.create_publisher(Image, '/lane/debug_image', 1)
        self.mask_pub = self.create_publisher(Image, '/lane/mask', 1)
        self.create_subscription(Image, image_topic, self.image_callback, qos_profile_sensor_data)
        self.create_service(DriveEnable, '/drive_enable', self.handle_drive_enable)
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info(
            f'LaneDrive ready: image={image_topic}, motor={motor_topic}, '
            f'enabled={self.enabled}. /drive_enable remains available for safety.'
        )

    def _declare_parameters(self):
        parameters = {
            'image_topic': '/image_raw',
            'motor_topic': '/control/lane',
            'rectified_topic': '/lane/rectified_image',
            'enable_fisheye_rectification': True,
            'calibration_file': '',
            'rectification_balance': 0.3,
            'auto_start': True,
            'max_speed': 10.0,
            'min_speed': 3.0,
            'speed_accel_limit': 8.0,
            'max_steering': 50.0,
            'kp': 0.16,
            'kd': 0.08,
            'error_filter_alpha': 0.35,
            'steering_sign': 1.0,
            'single_lane_speed_factor': 0.75,
            'avoidance_ready_max_error_px': 60.0,
            'handoff_ready_max_error_px': 40.0,
            'camera_timeout_sec': 0.5,
            'lost_lane_hold_sec': 0.1,
            'lost_lane_timeout_sec': 0.7,
            'roi_top_ratio': 0.52,
            'roi_bottom_ratio': 1.0,
            'roi_top_half_width_ratio': 0.28,
            'roi_bottom_margin_ratio': 0.02,
            'detection_bottom_ratio': 0.92,
            'camera_center_offset_px': 0,
            'lookahead_y_ratio': 0.72,
            'histogram_start_ratio': 0.72,
            'histogram_min_peak': 8,
            'white_min_saturation': 70,
            'white_min_value': 175,
            'yellow_hue_min': 15,
            'yellow_hue_max': 40,
            'yellow_min_saturation': 70,
            'yellow_min_value': 80,
            'blur_size': 5,
            'morphology_size': 5,
            'window_count': 9,
            'window_margin_px': 55,
            'window_min_pixels': 30,
            'minimum_lane_pixels': 160,
            'default_lane_width_ratio': 0.45,
            'minimum_lane_width_ratio': 0.22,
            'maximum_lane_width_ratio': 0.80,
            'lane_width_filter_alpha': 0.2,
            'yellow_side_min_pixels': 80,
            'yellow_side_deadband_px': 20,
        }
        for name, default in parameters.items():
            self.declare_parameter(name, default)

    def _load_fisheye_calibration(self):
        if not bool(self.get_parameter('enable_fisheye_rectification').value):
            self.get_logger().info('Fisheye rectification disabled by parameter')
            return

        configured_path = str(self.get_parameter('calibration_file').value)
        if configured_path:
            calibration_path = Path(configured_path).expanduser()
        else:
            calibration_path = Path(
                get_package_share_directory('lane_drive')
            ) / 'config' / 'fisheye_camera.yaml'

        try:
            with calibration_path.open('r', encoding='utf-8') as stream:
                calibration = yaml.safe_load(stream)
            if calibration.get('distortion_model') != 'equidistant':
                raise ValueError(
                    'distortion_model must be equidistant for fisheye correction'
                )
            self.camera_matrix = np.asarray(
                calibration['camera_matrix']['data'], dtype=np.float64
            ).reshape(3, 3)
            self.distortion = np.asarray(
                calibration['distortion_coefficients']['data'],
                dtype=np.float64,
            ).reshape(4, 1)
            self.calibration_size = (
                int(calibration['image_width']),
                int(calibration['image_height']),
            )
            self.rectification_enabled = True
            self.get_logger().info(
                f'Loaded fisheye calibration: {calibration_path} '
                f'({self.calibration_size[0]}x{self.calibration_size[1]})'
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
            balance = float(self.get_parameter('rectification_balance').value)
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

    def handle_drive_enable(self, request, response):
        self.enabled = request.mode == DriveEnable.Request.LKAS_MAIN
        if not self.enabled:
            self.publish_stop()
            self._reset_control()
        response.success = True
        response.message = (
            f'DriveEnable mode={request.mode} -> '
            f"{'LANE DRIVE ON' if self.enabled else 'lane drive sleep'}"
        )
        self.get_logger().info(response.message)
        return response

    def _reset_control(self):
        self.filtered_error = 0.0
        self.previous_error = 0.0
        self.last_angle = 0.0
        self.last_lane_time = None

    def image_callback(self, msg: Image):
        now = time.monotonic()
        self.last_image_time = now
        try:
            raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'Could not convert camera image: {exc}')
            return

        image = self._rectify_image(raw_image)
        if self.rectified_pub.get_subscription_count() > 0:
            rectified_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            rectified_msg.header = msg.header
            self.rectified_pub.publish(rectified_msg)

        result = self.detector.detect(image)
        self.avoidance_ready_pub.publish(Bool(data=lane_ready_for_avoidance(
            result,
            image.shape[1],
            float(self.get_parameter('camera_center_offset_px').value),
            float(self.get_parameter('avoidance_ready_max_error_px').value),
        )))
        self.handoff_ready_pub.publish(Bool(data=lane_ready_for_handoff(
            result,
            image.shape[1],
            float(self.get_parameter('camera_center_offset_px').value),
            float(self.get_parameter('handoff_ready_max_error_px').value),
        )))
        self.yellow_side_pub.publish(Int8(data=int(result.yellow_side)))
        # yellow_side denotes the divider's image side.  If the divider is on
        # the left, the vehicle is in the right lane, and vice versa.
        # Lane IDs: -1=left, 0=unknown, +1=right.
        self.ego_lane_pub.publish(Int8(data=int(-result.yellow_side)))
        if result.valid:
            self.last_lane_time = now
            angle, speed, error = self._compute_control(image.shape[1], result)
            if self.enabled:
                speed = self._limit_speed_increase(speed, now)
                self.publish_motor(angle, speed)
            self.error_pub.publish(Float32(data=float(error)))
        else:
            angle, speed, error = self._handle_lost_lane(now)

        if self.debug_pub.get_subscription_count() > 0:
            debug = self._draw_debug(image, result, angle, speed, error)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
        if self.mask_pub.get_subscription_count() > 0:
            self.mask_pub.publish(self.bridge.cv2_to_imgmsg(result.mask, encoding='mono8'))

    def _compute_control(self, width: int, result: LaneResult):
        image_center = width * 0.5 + float(self.get_parameter('camera_center_offset_px').value)
        raw_error = float(result.lane_center_x - image_center)
        alpha = float(self.get_parameter('error_filter_alpha').value)
        self.filtered_error = alpha * raw_error + (1.0 - alpha) * self.filtered_error
        derivative = self.filtered_error - self.previous_error
        self.previous_error = self.filtered_error

        angle = (
            float(self.get_parameter('steering_sign').value)
            * (
                float(self.get_parameter('kp').value) * self.filtered_error
                + float(self.get_parameter('kd').value) * derivative
            )
        )
        max_steering = float(self.get_parameter('max_steering').value)
        angle = float(np.clip(angle, -max_steering, max_steering))

        max_speed = float(self.get_parameter('max_speed').value)
        min_speed = float(self.get_parameter('min_speed').value)
        steering_ratio = min(1.0, abs(angle) / max(1.0, max_steering))
        speed = max_speed - (max_speed - min_speed) * steering_ratio
        if result.reason != 'both lanes':
            speed *= float(self.get_parameter('single_lane_speed_factor').value)
            speed = max(min_speed, speed)
        speed = float(np.clip(speed, min_speed, max_speed))
        self.last_angle = angle
        self.stop_published = False
        return angle, speed, raw_error

    def _limit_speed_increase(self, target_speed: float, now: float) -> float:
        """Apply the configured rise-rate limit to a positive speed command."""
        if self.last_speed_command_time is None:
            elapsed_sec = 0.0
        else:
            elapsed_sec = now - self.last_speed_command_time
        return limit_speed_increase(
            target_speed=target_speed,
            previous_speed=self.last_commanded_speed,
            elapsed_sec=elapsed_sec,
            acceleration_limit=float(self.get_parameter('speed_accel_limit').value),
            start_speed=float(self.get_parameter('min_speed').value),
        )

    def _handle_lost_lane(self, now: float):
        if self.last_lane_time is None:
            if self.enabled:
                self.publish_stop()
            return 0.0, 0.0, 0.0

        lost_for = now - self.last_lane_time
        hold_sec = float(self.get_parameter('lost_lane_hold_sec').value)
        timeout_sec = float(self.get_parameter('lost_lane_timeout_sec').value)
        if self.enabled and lost_for <= hold_sec:
            speed = float(self.get_parameter('min_speed').value)
            self.publish_motor(self.last_angle, speed)
            return self.last_angle, speed, self.filtered_error

        if self.enabled:
            self.publish_stop()
        if lost_for >= timeout_sec:
            self.get_logger().warn(
                f'Lane lost for {lost_for:.2f}s; motor stopped',
                throttle_duration_sec=1.0,
            )
        return 0.0, 0.0, self.filtered_error

    def watchdog_callback(self):
        if not self.enabled:
            return
        now = time.monotonic()
        timeout = float(self.get_parameter('camera_timeout_sec').value)
        if self.last_image_time is None or now - self.last_image_time > timeout:
            self.publish_stop()
            self.get_logger().warn(
                'Camera timeout; motor stopped', throttle_duration_sec=1.0
            )

    def publish_motor(self, angle: float, speed: float):
        msg = Float32MultiArray()
        msg.data = [float(angle), float(speed)]
        self.motor_pub.publish(msg)
        self.last_commanded_speed = max(0.0, float(speed))
        self.last_speed_command_time = time.monotonic()
        self.stop_published = False

    def publish_stop(self):
        if self.stop_published:
            return
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0]
        self.motor_pub.publish(msg)
        self.last_commanded_speed = 0.0
        self.last_speed_command_time = time.monotonic()
        self.stop_published = True

    @staticmethod
    def _draw_fit(image, fit, color, max_y=None):
        if fit is None:
            return
        height, width = image.shape[:2]
        draw_height = height if max_y is None else min(height, int(max_y))
        ys = np.arange(max(0, draw_height), dtype=np.int32)
        xs = np.polyval(fit, ys).astype(np.int32)
        valid = (xs >= 0) & (xs < width)
        points = np.column_stack((xs[valid], ys[valid])).reshape((-1, 1, 2))
        if len(points) >= 2:
            cv2.polylines(image, [points], False, color, 4)

    def _draw_debug(self, image, result, angle, speed, error):
        debug = image.copy()
        colored_mask = np.zeros_like(debug)
        colored_mask[:, :, 1] = result.mask
        debug = cv2.addWeighted(debug, 1.0, colored_mask, 0.25, 0.0)
        roi_polygon = self.detector.roi_polygon(*debug.shape[:2])
        cv2.polylines(debug, roi_polygon, True, (255, 255, 0), 3)
        for point in roi_polygon[0]:
            cv2.circle(debug, tuple(point), 6, (255, 255, 0), -1)
        cv2.putText(
            debug,
            'ROI',
            tuple(roi_polygon[0][2] + np.array([8, 22])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )
        height, width = debug.shape[:2]
        detection_bottom_y = self.detector.detection_bottom_y(height)
        if detection_bottom_y < height:
            cv2.line(
                debug,
                (0, detection_bottom_y),
                (width - 1, detection_bottom_y),
                (0, 0, 255),
                2,
            )
            cv2.putText(
                debug,
                'DETECTION BOTTOM',
                (10, max(20, detection_bottom_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
        self._draw_fit(
            debug, result.left_fit, (255, 100, 0), detection_bottom_y
        )
        self._draw_fit(
            debug, result.right_fit, (0, 200, 255), detection_bottom_y
        )
        camera_center = int(width * 0.5 + int(self.get_parameter('camera_center_offset_px').value))
        cv2.line(
            debug,
            (camera_center, height - 1),
            (camera_center, result.lookahead_y),
            (0, 0, 255),
            2,
        )
        if result.lane_center_x is not None:
            target = (int(result.lane_center_x), result.lookahead_y)
            cv2.circle(debug, target, 8, (255, 0, 255), -1)
            cv2.line(debug, (camera_center, result.lookahead_y), target, (255, 0, 255), 2)
        status = 'ON' if self.enabled else 'OFF'
        cv2.putText(
            debug,
            f'{status} {result.reason} conf={result.confidence:.2f}',
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.putText(
            debug,
            f'error={error:.1f}px angle={angle:.1f} speed={speed:.1f}',
            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
        )
        lane_states = (
            ('LEFT LANE', result.left_fit, 90),
            ('RIGHT LANE', result.right_fit, 120),
        )
        for label, fit, y_position in lane_states:
            detected = fit is not None
            state = 'DETECTED' if detected else 'MISSING'
            color = (0, 255, 0) if detected else (0, 0, 255)
            cv2.putText(
                debug,
                f'{label}: {state}',
                (15, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
        yellow_text = {
            -1: 'YELLOW: LEFT',
            0: 'YELLOW: UNKNOWN',
            1: 'YELLOW: RIGHT',
        }[result.yellow_side]
        yellow_color = (0, 255, 255) if result.yellow_side else (0, 0, 255)
        cv2.putText(
            debug,
            f'{yellow_text} conf={result.yellow_confidence:.2f}',
            (15, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            yellow_color,
            2,
        )
        return debug


def main(args=None):
    # Keep the ROS context alive until the final zero-speed command is sent.
    # rclpy's default SIGINT handler shuts the context down before ``finally``.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def stop_on_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    node = LaneDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_stop()
            # Allow the reliable DDS writer a brief chance to send the stop.
            time.sleep(0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == '__main__':
    main()
