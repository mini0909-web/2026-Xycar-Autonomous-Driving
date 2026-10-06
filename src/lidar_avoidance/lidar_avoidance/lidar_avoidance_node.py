#!/usr/bin/env python3
"""Arbitrate lane-drive commands and perform LiDAR-triggered lane changes."""

from enum import IntEnum
import math
import signal
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int8, String
from std_srvs.srv import Trigger

from .avoidance_core import (
    lane_change_bias,
    scan_to_points,
    summarize_box,
    ZoneSummary,
)


class AvoidanceState(IntEnum):
    """High-level obstacle avoidance states."""

    LANE_FOLLOW = 0
    WAIT_CLEAR = 1
    CHANGE_OUT = 2
    PASS_OBSTACLE = 3
    CHANGE_BACK = 4
    COOLDOWN = 5
    EMERGENCY_STOP = 6


SIDE_LEFT = -1
SIDE_RIGHT = 1


class LidarAvoidanceNode(Node):
    """Single final motor publisher for camera lane drive and LiDAR avoidance."""

    def __init__(self):
        super().__init__('lidar_avoidance')
        self._declare_parameters()

        self.state = AvoidanceState.LANE_FOLLOW
        self.state_started_at = time.monotonic()
        self.change_side = 0
        self.obstacle_confirm_count = 0
        self.front_clear_count = 0
        self.pass_clear_count = 0
        self.side_obstacle_seen = False

        self.points = []
        self.last_scan_time: Optional[float] = None
        self.base_angle = 0.0
        self.base_speed = 0.0
        self.last_lane_command_time: Optional[float] = None
        self.last_yellow_side = 0
        self.last_yellow_side_time: Optional[float] = None
        self.yellow_side_sequence = 0
        self.lane_confirmation_count = 0
        self.last_confirmation_sequence = -1
        self.stop_sent = False

        scan_topic = str(self._p('scan_topic'))
        lane_command_topic = str(self._p('lane_command_topic'))
        motor_topic = str(self._p('motor_topic'))
        yellow_side_topic = str(self._p('yellow_side_topic'))

        self.motor_pub = self.create_publisher(Float32MultiArray, motor_topic, 10)
        self.state_pub = self.create_publisher(String, '/lidar_avoidance/state', 10)
        self.debug_pub = self.create_publisher(
            Float32MultiArray, '/lidar_avoidance/debug', 10
        )
        self.create_subscription(
            LaserScan, scan_topic, self._scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Float32MultiArray, lane_command_topic, self._lane_command_callback, 10
        )
        self.create_subscription(Int8, yellow_side_topic, self._yellow_side_callback, 10)
        self.create_service(Trigger, '/lidar_avoidance/reset', self._reset_callback)
        self.control_timer = self.create_timer(
            float(self._p('control_period_sec')), self._control_callback
        )

        self.get_logger().info(
            'LidarAvoidance ready: '
            f'scan={scan_topic}, lane_cmd={lane_command_topic}, motor={motor_topic}. '
            'This node is the only node that should publish the final motor topic.'
        )

    def _declare_parameters(self):
        defaults = {
            'scan_topic': '/scan',
            'lane_command_topic': '/lane/drive_cmd',
            'motor_topic': '/xycar_motor',
            'yellow_side_topic': '/lane/yellow_side',
            'enabled': True,
            'control_period_sec': 0.05,
            'scan_timeout_sec': 0.35,
            'lane_command_timeout_sec': 0.35,
            'yellow_side_timeout_sec': 1.0,
            'scan_y_sign': 1.0,
            'minimum_valid_range_m': 0.12,
            'maximum_valid_range_m': 3.0,
            'front_x_min_m': 0.20,
            'front_x_max_m': 1.50,
            'front_half_width_m': 0.22,
            'front_min_points': 3,
            'obstacle_confirm_scans': 3,
            'front_clear_scans': 4,
            'emergency_distance_m': 0.30,
            'minimum_change_start_distance_m': 0.45,
            'target_lane_offset_m': 0.45,
            'target_half_width_m': 0.20,
            'target_x_min_m': -0.25,
            'target_x_max_m': 1.40,
            'target_blocked_min_points': 3,
            'original_lane_x_min_m': -0.40,
            'original_lane_x_max_m': 1.60,
            'original_lane_half_width_m': 0.25,
            'side_obstacle_min_points': 3,
            'lane_change_duration_sec': 1.20,
            'lane_change_confirmation_timeout_sec': 2.0,
            'lane_side_confirm_messages': 3,
            'lane_change_steering_bias': 22.0,
            'right_steering_sign': 1.0,
            'maximum_steering': 50.0,
            'avoidance_speed_cap': 8.0,
            'passing_min_sec': 0.80,
            'passing_timeout_sec': 4.0,
            'passing_clear_scans': 4,
            'cooldown_sec': 1.5,
            'allow_unknown_lane_side': False,
            'unknown_lane_fallback_side': 'left',
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _p(self, name):
        return self.get_parameter(name).value

    def _scan_callback(self, msg: LaserScan):
        configured_max = float(self._p('maximum_valid_range_m'))
        message_max = float(msg.range_max)
        maximum = min(configured_max, message_max) if message_max > 0.0 else configured_max
        minimum = max(float(self._p('minimum_valid_range_m')), float(msg.range_min))
        self.points = scan_to_points(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            minimum,
            maximum,
            float(self._p('scan_y_sign')),
        )
        self.last_scan_time = time.monotonic()

    def _lane_command_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn(
                'Ignored lane command with fewer than two values',
                throttle_duration_sec=1.0,
            )
            return
        angle = float(msg.data[0])
        speed = float(msg.data[1])
        if not (math.isfinite(angle) and math.isfinite(speed)):
            return
        self.base_angle = angle
        self.base_speed = max(0.0, speed)
        self.last_lane_command_time = time.monotonic()

    def _yellow_side_callback(self, msg: Int8):
        side = int(msg.data)
        if side in (SIDE_LEFT, SIDE_RIGHT):
            self.last_yellow_side = side
            self.last_yellow_side_time = time.monotonic()
            self.yellow_side_sequence += 1

    def _zone_summaries(self) -> Dict[str, ZoneSummary]:
        front = summarize_box(
            self.points,
            float(self._p('front_x_min_m')),
            float(self._p('front_x_max_m')),
            -float(self._p('front_half_width_m')),
            float(self._p('front_half_width_m')),
        )
        summaries = {'front': front}
        for name, side in (('left', SIDE_LEFT), ('right', SIDE_RIGHT)):
            # Standard LaserScan coordinates use +y for the vehicle's left.
            target_center_y = -side * float(self._p('target_lane_offset_m'))
            half_width = float(self._p('target_half_width_m'))
            summaries[name] = summarize_box(
                self.points,
                float(self._p('target_x_min_m')),
                float(self._p('target_x_max_m')),
                target_center_y - half_width,
                target_center_y + half_width,
            )
        if self.change_side in (SIDE_LEFT, SIDE_RIGHT):
            # After changing lanes, the original lane is opposite the change side.
            original_center_y = (
                self.change_side * float(self._p('target_lane_offset_m'))
            )
            half_width = float(self._p('original_lane_half_width_m'))
            summaries['original'] = summarize_box(
                self.points,
                float(self._p('original_lane_x_min_m')),
                float(self._p('original_lane_x_max_m')),
                original_center_y - half_width,
                original_center_y + half_width,
            )
        else:
            summaries['original'] = ZoneSummary(0, math.inf)
        return summaries

    def _front_blocked(self, summary: ZoneSummary) -> bool:
        return summary.point_count >= int(self._p('front_min_points'))

    def _target_blocked(self, summary: ZoneSummary) -> bool:
        return summary.point_count >= int(self._p('target_blocked_min_points'))

    def _valid_change_side(self, now: float) -> int:
        if (
            self.last_yellow_side_time is not None
            and now - self.last_yellow_side_time
            <= float(self._p('yellow_side_timeout_sec'))
        ):
            return self.last_yellow_side
        if not bool(self._p('allow_unknown_lane_side')):
            return 0
        fallback = str(self._p('unknown_lane_fallback_side')).strip().lower()
        return SIDE_RIGHT if fallback == 'right' else SIDE_LEFT

    @staticmethod
    def _summary_for_side(summaries, side: int) -> ZoneSummary:
        return summaries['left' if side == SIDE_LEFT else 'right']

    def _transition(self, state: AvoidanceState, reason: str):
        if state == self.state:
            return
        previous = self.state
        self.state = state
        self.state_started_at = time.monotonic()
        self.front_clear_count = 0
        if state == AvoidanceState.CHANGE_OUT:
            self.side_obstacle_seen = False
            self._reset_lane_confirmation()
        if state == AvoidanceState.CHANGE_BACK:
            self._reset_lane_confirmation()
        if state == AvoidanceState.PASS_OBSTACLE:
            self.pass_clear_count = 0
        if state in (AvoidanceState.LANE_FOLLOW, AvoidanceState.COOLDOWN):
            self.obstacle_confirm_count = 0
        self.get_logger().info(f'{previous.name} -> {state.name}: {reason}')

    def _control_callback(self):
        now = time.monotonic()
        if not self._inputs_fresh(now):
            self._publish_stop()
            self._publish_state(None, None, 0.0)
            return

        summaries = self._zone_summaries()
        front = summaries['front']
        front_blocked = self._front_blocked(front)

        if not bool(self._p('enabled')):
            self._publish_motor(self.base_angle, self.base_speed)
            self._publish_state(front, None, 0.0)
            return

        angle = self.base_angle
        speed = self.base_speed
        steering_bias = 0.0

        if self.state == AvoidanceState.LANE_FOLLOW:
            self.obstacle_confirm_count = (
                self.obstacle_confirm_count + 1 if front_blocked else 0
            )
            if self.obstacle_confirm_count >= int(self._p('obstacle_confirm_scans')):
                change_side = self._valid_change_side(now)
                if change_side == 0:
                    self._transition(
                        AvoidanceState.WAIT_CLEAR,
                        'obstacle confirmed but yellow divider side is unknown',
                    )
                else:
                    target = self._summary_for_side(summaries, change_side)
                    too_close = front.nearest_range < float(
                        self._p('minimum_change_start_distance_m')
                    )
                    if too_close or self._target_blocked(target):
                        reason = 'obstacle too close' if too_close else 'target lane blocked'
                        self._transition(AvoidanceState.WAIT_CLEAR, reason)
                    else:
                        self.change_side = change_side
                        self._transition(
                            AvoidanceState.CHANGE_OUT,
                            'front obstacle confirmed and target lane is clear',
                        )

        elif self.state == AvoidanceState.WAIT_CLEAR:
            speed = 0.0
            if front_blocked:
                self.front_clear_count = 0
                change_side = self._valid_change_side(now)
                if change_side != 0:
                    target = self._summary_for_side(summaries, change_side)
                    far_enough = front.nearest_range >= float(
                        self._p('minimum_change_start_distance_m')
                    )
                    if far_enough and not self._target_blocked(target):
                        self.change_side = change_side
                        self._transition(
                            AvoidanceState.CHANGE_OUT,
                            'target lane became clear',
                        )
            else:
                self.front_clear_count += 1
                if self.front_clear_count >= int(self._p('front_clear_scans')):
                    self.change_side = 0
                    self._transition(AvoidanceState.LANE_FOLLOW, 'front path is clear')

        elif self.state == AvoidanceState.CHANGE_OUT:
            target = self._summary_for_side(summaries, self.change_side)
            original = summaries['original']
            self.side_obstacle_seen = self.side_obstacle_seen or (
                original.point_count >= int(self._p('side_obstacle_min_points'))
            )
            if (
                self._target_blocked(target)
                and target.nearest_range <= float(self._p('emergency_distance_m'))
            ):
                self._transition(
                    AvoidanceState.EMERGENCY_STOP,
                    'close obstacle appeared in target lane',
                )
                speed = 0.0
            else:
                elapsed = now - self.state_started_at
                target_lane_confirmed = self._confirm_yellow_side(
                    -self.change_side
                )
                steering_bias = self._lane_change_bias(elapsed, self.change_side)
                speed = min(speed, float(self._p('avoidance_speed_cap')))
                if (
                    elapsed >= float(self._p('lane_change_duration_sec'))
                    and target_lane_confirmed
                ):
                    self._transition(
                        AvoidanceState.PASS_OBSTACLE, 'outbound lane change complete'
                    )
                elif elapsed >= float(
                    self._p('lane_change_confirmation_timeout_sec')
                ):
                    self._transition(
                        AvoidanceState.EMERGENCY_STOP,
                        'camera did not confirm arrival in the target lane',
                    )
                    speed = 0.0

        elif self.state == AvoidanceState.PASS_OBSTACLE:
            elapsed = now - self.state_started_at
            speed = min(speed, float(self._p('avoidance_speed_cap')))
            original = summaries['original']
            original_blocked = (
                original.point_count >= int(self._p('side_obstacle_min_points'))
            )
            self.side_obstacle_seen = self.side_obstacle_seen or original_blocked
            if not front_blocked and not original_blocked:
                self.pass_clear_count += 1
            else:
                self.pass_clear_count = 0

            if (
                elapsed >= float(self._p('passing_min_sec'))
                and self.side_obstacle_seen
                and self.pass_clear_count >= int(self._p('passing_clear_scans'))
            ):
                self._transition(
                    AvoidanceState.CHANGE_BACK,
                    'obstacle is behind and original lane is clear',
                )
            elif elapsed >= float(self._p('passing_timeout_sec')):
                self._transition(
                    AvoidanceState.EMERGENCY_STOP,
                    'could not verify a safe return before passing timeout',
                )
                speed = 0.0

        elif self.state == AvoidanceState.CHANGE_BACK:
            original = summaries['original']
            original_blocked = (
                original.point_count >= int(self._p('target_blocked_min_points'))
            )
            if (
                original_blocked
                and original.nearest_range <= float(self._p('emergency_distance_m'))
            ):
                self._transition(
                    AvoidanceState.EMERGENCY_STOP,
                    'original lane became blocked during return',
                )
                speed = 0.0
            else:
                elapsed = now - self.state_started_at
                original_lane_confirmed = self._confirm_yellow_side(
                    self.change_side
                )
                steering_bias = self._lane_change_bias(elapsed, -self.change_side)
                speed = min(speed, float(self._p('avoidance_speed_cap')))
                if (
                    elapsed >= float(self._p('lane_change_duration_sec'))
                    and original_lane_confirmed
                ):
                    self._transition(AvoidanceState.COOLDOWN, 'return lane change complete')
                elif elapsed >= float(
                    self._p('lane_change_confirmation_timeout_sec')
                ):
                    self._transition(
                        AvoidanceState.EMERGENCY_STOP,
                        'camera did not confirm return to the original lane',
                    )
                    speed = 0.0

        elif self.state == AvoidanceState.COOLDOWN:
            if now - self.state_started_at >= float(self._p('cooldown_sec')):
                self.change_side = 0
                self._transition(AvoidanceState.LANE_FOLLOW, 'cooldown complete')

        elif self.state == AvoidanceState.EMERGENCY_STOP:
            speed = 0.0

        angle = self._clamp_steering(angle + steering_bias)
        if speed <= 0.0:
            self._publish_stop(angle)
        else:
            self._publish_motor(angle, speed)

        target = None
        if self.change_side in (SIDE_LEFT, SIDE_RIGHT):
            target = self._summary_for_side(summaries, self.change_side)
        self._publish_state(front, target, steering_bias)

    def _inputs_fresh(self, now: float) -> bool:
        if self.last_lane_command_time is None:
            self.get_logger().warn(
                'No lane command received; motor stopped', throttle_duration_sec=1.0
            )
            return False
        if now - self.last_lane_command_time > float(self._p('lane_command_timeout_sec')):
            self.get_logger().warn(
                'Lane command timeout; motor stopped', throttle_duration_sec=1.0
            )
            return False
        if self.last_scan_time is None:
            self.get_logger().warn(
                'No LiDAR scan received; motor stopped', throttle_duration_sec=1.0
            )
            return False
        if now - self.last_scan_time > float(self._p('scan_timeout_sec')):
            self.get_logger().warn(
                'LiDAR scan timeout; motor stopped', throttle_duration_sec=1.0
            )
            return False
        return True

    def _lane_change_bias(self, elapsed: float, physical_side: int) -> float:
        steering_direction = (
            physical_side * float(self._p('right_steering_sign'))
        )
        return lane_change_bias(
            elapsed,
            float(self._p('lane_change_duration_sec')),
            steering_direction,
            float(self._p('lane_change_steering_bias')),
        )

    def _reset_lane_confirmation(self):
        self.lane_confirmation_count = 0
        self.last_confirmation_sequence = self.yellow_side_sequence

    def _confirm_yellow_side(self, expected_side: int) -> bool:
        """Require multiple new camera messages before confirming a lane."""
        if self.yellow_side_sequence == self.last_confirmation_sequence:
            return self.lane_confirmation_count >= int(
                self._p('lane_side_confirm_messages')
            )
        self.last_confirmation_sequence = self.yellow_side_sequence
        if self.last_yellow_side == expected_side:
            self.lane_confirmation_count += 1
        else:
            self.lane_confirmation_count = 0
        return self.lane_confirmation_count >= int(
            self._p('lane_side_confirm_messages')
        )

    def _clamp_steering(self, angle: float) -> float:
        maximum = abs(float(self._p('maximum_steering')))
        return min(maximum, max(-maximum, float(angle)))

    def _publish_motor(self, angle: float, speed: float):
        msg = Float32MultiArray()
        msg.data = [float(angle), max(0.0, float(speed))]
        self.motor_pub.publish(msg)
        self.stop_sent = False

    def _publish_stop(self, angle: float = 0.0):
        # Keep publishing stop when the supervisor is active so a stale command
        # cannot remain at the ROS1 bridge/VESC side.
        msg = Float32MultiArray()
        msg.data = [self._clamp_steering(angle), 0.0]
        self.motor_pub.publish(msg)
        self.stop_sent = True

    def _publish_state(self, front, target, steering_bias: float):
        self.state_pub.publish(String(data=self.state.name))
        front_count = float(front.point_count) if front is not None else 0.0
        front_range = (
            float(front.nearest_range)
            if front is not None and math.isfinite(front.nearest_range)
            else -1.0
        )
        target_count = float(target.point_count) if target is not None else 0.0
        target_range = (
            float(target.nearest_range)
            if target is not None and math.isfinite(target.nearest_range)
            else -1.0
        )
        msg = Float32MultiArray()
        msg.data = [
            float(self.state),
            front_count,
            front_range,
            target_count,
            target_range,
            float(steering_bias),
            float(self.last_yellow_side),
        ]
        self.debug_pub.publish(msg)

    def _reset_callback(self, _request, response):
        now = time.monotonic()
        if not self._inputs_fresh(now):
            response.success = False
            response.message = 'Cannot reset while LiDAR or lane command is missing'
            return response
        front = self._zone_summaries()['front']
        if self._front_blocked(front):
            response.success = False
            response.message = 'Cannot reset while the front corridor is blocked'
            return response
        self.change_side = 0
        self._transition(AvoidanceState.LANE_FOLLOW, 'manual reset')
        response.success = True
        response.message = 'Avoidance supervisor reset to LANE_FOLLOW'
        return response


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def stop_on_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    node = LidarAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._publish_stop()
            time.sleep(0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == '__main__':
    main()
