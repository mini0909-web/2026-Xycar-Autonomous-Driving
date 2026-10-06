#!/usr/bin/env python3
"""LiDAR-centerline-based obstacle avoidance with an E2E handoff."""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import List, Optional, Sequence, Tuple

from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray, Int8, String
from visualization_msgs.msg import Marker, MarkerArray


# Obstacle-side values published on /obstacle/lane for compatibility.
# LiDAR coordinates are x=forward, y=left/right: positive y is LEFT.
LEFT, UNKNOWN, RIGHT = 1, 0, -1
NONE = 'NONE'


class AvoidanceState(str, Enum):
    E2E_FOLLOW = 'E2E_FOLLOW'
    AVOIDING = 'AVOIDING'

    # Legacy lane-perception states (disabled; retained as documentation):
    # LANE_FOLLOW = 'LANE_FOLLOW'
    # CHANGE_OUT = 'CHANGE_OUT'
    # RECENTER = 'RECENTER'
    # COOLDOWN = 'COOLDOWN'


@dataclass
class ObstacleCluster:
    points: List[Tuple[float, float]]

    @property
    def x(self) -> float:
        return sum(p[0] for p in self.points) / len(self.points)

    @property
    def y(self) -> float:
        return sum(p[1] for p in self.points) / len(self.points)

    @property
    def distance(self) -> float:
        return min(math.hypot(*p) for p in self.points)

    @property
    def width(self) -> float:
        return max(p[1] for p in self.points) - min(p[1] for p in self.points)

    @property
    def depth(self) -> float:
        return max(p[0] for p in self.points) - min(p[0] for p in self.points)


class DynamicObstacleAvoidance(Node):
    """Steer away from the closest front obstacle, then release to E2E."""

    def __init__(self) -> None:
        super().__init__('dynamic_obstacle_avoidance')
        self._declare_parameters()
        lidar_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.motor_pub = self.create_publisher(Float32MultiArray, self._p('motor_topic'), 10)
        self.active_pub = self.create_publisher(Bool, self._p('active_topic'), 10)
        self.obstacle_lane_pub = self.create_publisher(Int8, self._p('obstacle_lane_topic'), 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', 10)
        self.create_subscription(LaserScan, self._p('scan_topic'), self._scan_callback, lidar_qos)
        self.create_subscription(
            Float32MultiArray, self._p('e2e_command_topic'), self._e2e_command_callback, 10)
        self.create_subscription(
            Bool, self._p('cone_active_topic'), self._cone_active_callback, 10)

        # Legacy camera-lane gates are intentionally disabled. E2E /motor_lane is
        # still subscribed because it remains the mux fallback after avoidance.
        # self.create_subscription(
        #     Int8, self._p('ego_lane_topic'), self._ego_lane_callback, 10)
        # self.create_subscription(
        #     Bool, self._p('lane_ready_topic'), self._lane_ready_callback, 10)
        # self.create_subscription(
        #     Bool, self._p('handoff_ready_topic'), self._handoff_ready_callback, 10)

        self.latest_clusters: List[ObstacleCluster] = []
        self.scan_received = False
        self.e2e_command: Optional[Tuple[float, float]] = None
        self.target: Optional[ObstacleCluster] = None
        self.locked_target: Optional[ObstacleCluster] = None
        self.target_matched = False
        self.avoidance_started_at: Optional[float] = None
        self.target_side = NONE
        self.avoidance_direction = NONE
        self.obstacle_confirm_count = 0
        self.obstacle_lost_count = 0
        self.state = AvoidanceState.E2E_FOLLOW
        self.cone_active = False
        self.last_logged_state = None

        # Disabled lane-perception state retained for future reference:
        # self.ego_lane = UNKNOWN_LANE
        # self.lane_ready = False
        # self.lane_ready_received_at = None
        # self.handoff_ready = False
        # self.handoff_ready_received_at = None
        # self.origin_lane = UNKNOWN_LANE
        # self.lane_change_confirm_count = 0
        # self.handoff_confirm_count = 0

        self.timer = self.create_timer(1.0 / self._p('control_rate_hz'), self._control_callback)
        self.get_logger().info(
            'Dynamic avoidance ready: LiDAR centerline gated E2E override')

    def _declare_parameters(self) -> None:
        self.declare_parameters('', [
            ('scan_topic', '/scan'), ('marker_frame', 'laser_frame'),
            ('motor_topic', '/cmd/dynamic'), ('active_topic', '/dynamic/active'),
            ('e2e_command_topic', '/motor_lane'),
            ('cone_active_topic', '/cone/active'),
            ('obstacle_lane_topic', '/obstacle/lane'),
            ('control_rate_hz', 30.0),
            ('lidar_yaw_offset_deg', 0.0), ('cluster_gap_m', 0.2),
            ('lidar_detection_half_angle_deg', 110.0),
            ('obstacle_detection_half_angle_deg', 90.0),
            ('min_cluster_points', 3), ('detection_distance_m', 1.0),
            ('detection_half_width_m', 0.4),
            ('front_detection_distance_m', 0.8), ('front_detection_half_width_m', 0.17),
            ('front_detection_min_points', 3), ('obstacle_confirm_scans', 1),
            ('obstacle_lost_scans', 5),
            ('target_tracking_distance_m', 0.05),
            ('max_avoidance_duration_sec', 2.0),
            ('avoidance_steering_angle', 25.0),
            ('avoidance_speed', 15), ('max_steering', 30.0), ('steering_sign', 1.0),
            # Disabled lane-perception parameters retained for reference:
            # ('ego_lane_topic', '/lane/ego_lane'),
            # ('lane_ready_topic', '/lane/avoidance_ready'),
            # ('lane_ready_timeout_sec', 0.25),
            # ('handoff_ready_topic', '/lane/handoff_ready'),
            # ('target_lane_clearance_m', 0.1),
            # ('lane_change_steering_bias', 40.0),
            # ('lane_change_confirmation_scans', 10),
            # ('minimum_lane_change_duration_sec', 2.0),
            # ('recenter_duration_sec', 0.4),
            # ('handoff_confirmation_frames', 3),
            # ('lane_change_confirmation_timeout_sec', 5.0),
            # ('cooldown_sec', 1.0),
        ])

    def _p(self, name: str):
        return self.get_parameter(name).value

    def _scan_callback(self, scan: LaserScan) -> None:
        self.scan_received = True
        points = []
        yaw_offset = math.radians(self._p('lidar_yaw_offset_deg'))
        for i, distance in enumerate(scan.ranges):
            if (not math.isfinite(distance)
                    or distance < scan.range_min or distance > scan.range_max):
                continue
            angle = scan.angle_min + i * scan.angle_increment + yaw_offset
            angle = math.atan2(math.sin(angle), math.cos(angle))
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            if self._point_in_detection_roi(x, y):
                points.append((x, y))
        self.latest_clusters = self._cluster_points(points)
        released = self._update_obstacle_observation()
        if released:
            # Publish inactive from the scan callback so the third missed scan
            # releases the mux immediately, rather than waiting for another scan.
            self._publish(self.state.value, False, *self._base_command())

    def _point_in_detection_roi(self, x: float, y: float) -> bool:
        """Keep the existing LiDAR clustering ROI unchanged."""
        angle_deg = abs(math.degrees(math.atan2(y, x)))
        return (
            x <= float(self._p('detection_distance_m'))
            and abs(y) <= float(self._p('detection_half_width_m'))
            and angle_deg <= float(self._p('lidar_detection_half_angle_deg'))
        )

    def _cluster_points(self, points: Sequence[Tuple[float, float]]) -> List[ObstacleCluster]:
        clusters, current = [], []
        for point in points:
            if current and math.dist(current[-1], point) > self._p('cluster_gap_m'):
                if len(current) >= self._p('min_cluster_points'):
                    clusters.append(ObstacleCluster(current))
                current = []
            current.append(point)
        if len(current) >= self._p('min_cluster_points'):
            clusters.append(ObstacleCluster(current))
        return clusters

    def _valid_front_cluster(self, cluster: ObstacleCluster) -> bool:
        angle_deg = abs(math.degrees(math.atan2(cluster.y, cluster.x)))
        return (
            angle_deg <= float(self._p('obstacle_detection_half_angle_deg'))
            and cluster.x <= float(self._p('front_detection_distance_m'))
            and abs(cluster.y) <= float(self._p('front_detection_half_width_m'))
            and cluster.width < 0.50
            and cluster.depth < 0.50
            and len(cluster.points) >= int(self._p('front_detection_min_points'))
        )

    def _front_candidates(self) -> List[ObstacleCluster]:
        return [c for c in self.latest_clusters if self._valid_front_cluster(c)]

    def _select_new_target(self) -> Optional[ObstacleCluster]:
        candidates = self._front_candidates()
        return min(candidates, key=lambda c: c.distance) if candidates else None

    def _front_target(self) -> Optional[ObstacleCluster]:
        """Compatibility wrapper for callers that need a new front target."""
        return self._select_new_target()

    def _match_locked_target(self) -> Optional[ObstacleCluster]:
        """Match only the previously locked target; never acquire a replacement."""
        if self.locked_target is None:
            return None
        candidates = self._front_candidates()
        if not candidates:
            return None
        closest = min(
            candidates,
            key=lambda c: math.hypot(
                c.x - self.locked_target.x, c.y - self.locked_target.y),
        )
        separation = math.hypot(
            closest.x - self.locked_target.x,
            closest.y - self.locked_target.y,
        )
        if separation > float(self._p('target_tracking_distance_m')):
            return None
        return closest

    def _classify_target_side(self, target: Optional[ObstacleCluster]) -> str:
        if target is None:
            return NONE
        # Split every detected target directly on the LiDAR forward centerline.
        # A target exactly on y=0 belongs to LEFT, so a detected target always
        # has one of the two actionable sides.
        if target.y >= 0.0:
            return 'LEFT'
        return 'RIGHT'

    def _direction_away_from(self, target_side: str) -> str:
        if target_side == 'LEFT':
            return 'RIGHT'
        if target_side == 'RIGHT':
            return 'LEFT'
        return NONE

    def _update_obstacle_observation(self) -> bool:
        """Update counters once per LaserScan; return True on E2E release."""
        if self.state == AvoidanceState.AVOIDING:
            matched = self._match_locked_target()
            self.target_matched = matched is not None
            if matched is not None:
                # LaserScan coordinates are vehicle-relative, so advance the
                # track with every match instead of retaining its first pose.
                self.locked_target = matched
                self.target = matched
                self.obstacle_lost_count = 0
            else:
                # Other visible clusters must not replace the locked target.
                self.target = self.locked_target
                self.obstacle_lost_count += 1

            side_value = {'LEFT': LEFT, 'RIGHT': RIGHT}.get(self.target_side, UNKNOWN)
            self.obstacle_lane_pub.publish(Int8(data=side_value))
            if self.obstacle_lost_count < int(self._p('obstacle_lost_scans')):
                return False
            self._release_target('locked target lost for configured scans')
            return True

        target = self._select_new_target()
        self.target = target
        self.target_matched = target is not None
        self.target_side = self._classify_target_side(target)
        side_value = {'LEFT': LEFT, 'RIGHT': RIGHT}.get(self.target_side, UNKNOWN)
        self.obstacle_lane_pub.publish(Int8(data=side_value))

        if target is not None:
            self.obstacle_lost_count = 0
            self.obstacle_confirm_count += 1
            return False

        self.obstacle_confirm_count = 0
        self.obstacle_lost_count = 0
        return False

    def _e2e_command_callback(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 2:
            self.e2e_command = float(msg.data[0]), float(msg.data[1])

    def _cone_active_callback(self, msg: Bool) -> None:
        self.cone_active = bool(msg.data)
        if self.cone_active and self.state == AvoidanceState.AVOIDING:
            self._release_target('cone mode took priority')

    # Legacy camera-lane callbacks and gates are disabled. The original logic is
    # retained below as comments, per the migration requirement.
    #
    # def _ego_lane_callback(self, msg: Int8) -> None:
    #     lane = int(msg.data)
    #     self.ego_lane = lane if lane in (LEFT_LANE, RIGHT_LANE) else UNKNOWN_LANE
    #     if self.state == AvoidanceState.CHANGE_OUT:
    #         if self.ego_lane == -self.origin_lane:
    #             self.lane_change_confirm_count += 1
    #         else:
    #             self.lane_change_confirm_count = 0
    #
    # def _lane_ready_callback(self, msg: Bool) -> None:
    #     self.lane_ready = bool(msg.data)
    #     self.lane_ready_received_at = time.monotonic()
    #
    # def _handoff_ready_callback(self, msg: Bool) -> None:
    #     self.handoff_ready = bool(msg.data)
    #     self.handoff_ready_received_at = time.monotonic()
    #     if self.state == AvoidanceState.RECENTER:
    #         if self.handoff_ready:
    #             self.handoff_confirm_count += 1
    #         else:
    #             self.handoff_confirm_count = 0
    #
    # def _handoff_is_fresh(self) -> bool:
    #     return (self.handoff_ready
    #             and self.handoff_ready_received_at is not None
    #             and time.monotonic() - self.handoff_ready_received_at
    #             <= float(self._p('lane_ready_timeout_sec')))
    #
    # def _lane_is_ready_for_avoidance(self) -> bool:
    #     return (self.lane_ready
    #             and self.lane_ready_received_at is not None
    #             and time.monotonic() - self.lane_ready_received_at
    #             <= float(self._p('lane_ready_timeout_sec')))
    #
    # def _same_lane_obstacle_detected(self) -> bool:
    #     return (self._lane_is_ready_for_avoidance()
    #             and self.ego_lane in (LEFT_LANE, RIGHT_LANE)
    #             and self.classified_obstacle_lane == self.ego_lane
    #             and self.target is not None
    #             and self.obstacle_confirm_count
    #             >= int(self._p('obstacle_confirm_scans')))
    #
    # Disabled state-machine conditions:
    # CHANGE_OUT waited for minimum_lane_change_duration_sec plus repeated
    # opposite ego_lane messages, then transitioned to RECENTER.
    # RECENTER waited for recenter_duration_sec plus fresh repeated
    # /lane/handoff_ready messages, then transitioned to COOLDOWN.
    # COOLDOWN waited for cooldown_sec before returning to LANE_FOLLOW.

    def _transition(self, state: AvoidanceState, reason: str) -> None:
        if state == self.state:
            return
        self.get_logger().info('%s -> %s: %s' % (self.state.value, state.value, reason))
        self.state = state

    def _reset_avoidance(self) -> None:
        self.target = None
        self.locked_target = None
        self.target_matched = False
        self.avoidance_started_at = None
        self.target_side = NONE
        self.avoidance_direction = NONE
        self.obstacle_confirm_count = 0
        self.obstacle_lost_count = 0

    def _lock_target(self, target: ObstacleCluster) -> None:
        self.locked_target = target
        self.target = target
        self.target_matched = True
        self.target_side = self._classify_target_side(target)
        self.avoidance_direction = self._direction_away_from(self.target_side)
        self.obstacle_lost_count = 0
        self.avoidance_started_at = time.monotonic()

    def _release_target(self, reason: str) -> None:
        self._transition(AvoidanceState.E2E_FOLLOW, reason)
        self._reset_avoidance()

    def _avoidance_timed_out(self) -> bool:
        duration = float(self._p('max_avoidance_duration_sec'))
        return (
            duration > 0.0
            and self.avoidance_started_at is not None
            and time.monotonic() - self.avoidance_started_at >= duration
        )

    def _base_command(self) -> Tuple[float, float]:
        return self.e2e_command if self.e2e_command is not None else (0.0, 0.0)

    def _avoidance_command(self) -> Tuple[float, float]:
        # Vehicle convention retained from the existing implementation/tests:
        # negative steering=left, positive steering=right.
        direction_sign = -1.0 if self.avoidance_direction == 'LEFT' else 1.0
        angle = (
            float(self._p('steering_sign'))
            * direction_sign
            * abs(float(self._p('avoidance_steering_angle')))
        )
        max_steering = abs(float(self._p('max_steering')))
        angle = max(-max_steering, min(max_steering, angle))
        return angle, float(self._p('avoidance_speed'))

    def _control_callback(self) -> None:
        if not self.scan_received:
            self._publish('WAITING_FOR_LIDAR', False, 0.0, 0.0)
            return

        if self.state == AvoidanceState.E2E_FOLLOW:
            confirmed = self.obstacle_confirm_count >= int(self._p('obstacle_confirm_scans'))
            if not self.cone_active and self.target is not None and confirmed:
                self._lock_target(self.target)
                self._transition(AvoidanceState.AVOIDING, 'front obstacle confirmed')
                self._publish(self.state.value, True, *self._avoidance_command())
                return
            self._publish(self.state.value, False, *self._base_command())
            return

        if self.state == AvoidanceState.AVOIDING:
            if self._avoidance_timed_out():
                self._release_target('maximum avoidance duration exceeded')
                self._publish(self.state.value, False, *self._base_command())
                return
            self._publish(self.state.value, True, *self._avoidance_command())
            return

        self._publish(self.state.value, False, *self._base_command())

    def _status_text(self, status: str) -> str:
        return (
            f'state={status} target={self.target_side} '
            f'avoid={self.avoidance_direction} '
            f'target_locked={self.locked_target is not None} '
            f'target_matched={self.target_matched} '
            f'confirm={self.obstacle_confirm_count} lost={self.obstacle_lost_count}'
        )

    def _publish(self, status: str, active: bool, steering: float, speed: float) -> None:
        self.active_pub.publish(Bool(data=active))
        if active:
            self.motor_pub.publish(Float32MultiArray(data=[float(steering), float(speed)]))
        detailed_status = self._status_text(status)
        self.status_pub.publish(String(data=detailed_status))
        # RViz 시각화 꺼둠(경량화) - MarkerArray 생성 자체가 매 제어 주기마다
        # 도는 비용이라 안 부르는 쪽으로 뺐다.
        # self._publish_markers(detailed_status, steering)

    @staticmethod
    def _rviz_status_text(status: str) -> str:
        if status == 'WAITING_FOR_LIDAR':
            return 'WAITING FOR LIDAR'
        if status == AvoidanceState.AVOIDING.value:
            return 'OBSTACLE AVOIDANCE'
        return 'E2E FOLLOW'

    def _publish_markers(self, status: str, steering: float) -> None:
        markers = MarkerArray()
        markers.markers.append(Marker(action=Marker.DELETEALL))
        now = self.get_clock().now().to_msg()
        frame = self._p('marker_frame')
        for i, cluster in enumerate(self.latest_clusters):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = frame, now
            marker.ns, marker.id = 'clusters', i
            marker.type, marker.action = Marker.CUBE, Marker.ADD
            marker.pose.position.x, marker.pose.position.y = cluster.x, cluster.y
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = 0.12, max(cluster.width, 0.1), 0.2
            marker.color.a = 0.8
            marker.color.r, marker.color.g, marker.color.b = (
                (1.0, 0.1, 0.1) if cluster is self.target else (0.1, 0.8, 1.0))
            markers.markers.append(marker)

            width_marker = Marker()
            width_marker.header.frame_id, width_marker.header.stamp = frame, now
            width_marker.ns, width_marker.id = 'cluster_widths', i
            width_marker.type, width_marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            width_marker.pose.position.x = cluster.x
            width_marker.pose.position.y = cluster.y
            width_marker.pose.position.z = 0.25
            width_marker.pose.orientation.w = 1.0
            width_marker.scale.z = 0.12
            width_marker.color.r = 1.0
            width_marker.color.g = 1.0
            width_marker.color.b = 1.0
            width_marker.color.a = 1.0
            width_marker.text = f'{cluster.width * 100.0:.1f} cm'
            markers.markers.append(width_marker)

        centerline = Marker()
        centerline.header.frame_id, centerline.header.stamp = frame, now
        centerline.ns, centerline.id = 'centerline', 900
        centerline.type, centerline.action = Marker.LINE_STRIP, Marker.ADD
        centerline.scale.x = 0.025
        centerline.color.a, centerline.color.g, centerline.color.b = 1.0, 1.0, 1.0
        centerline.points = [Point(x=0.0, y=0.0), Point(
            x=float(self._p('detection_distance_m')), y=0.0)]
        markers.markers.append(centerline)

        arrow = Marker()
        arrow.header.frame_id, arrow.header.stamp = frame, now
        arrow.ns, arrow.id, arrow.type, arrow.action = 'command', 1000, Marker.ARROW, Marker.ADD
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.05, 0.1, 0.12
        arrow.color.a, arrow.color.g = 1.0, 1.0
        arrow.points = [Point(x=0.0, y=0.0), Point(
            x=1.0, y=-steering / max(1.0, abs(float(self._p('max_steering')))) * 0.5)]
        markers.markers.append(arrow)

        text_marker = Marker()
        text_marker.header.frame_id, text_marker.header.stamp = frame, now
        text_marker.ns, text_marker.id = 'avoidance_status', 2000
        text_marker.type, text_marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        text_marker.pose.position.x, text_marker.pose.position.z = 0.8, 0.7
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z, text_marker.color.a = 0.18, 1.0
        if self.state == AvoidanceState.AVOIDING:
            text_marker.color.r, text_marker.color.g = 1.0, 0.65
        else:
            text_marker.color.g, text_marker.color.b = 0.8, 1.0
        text_marker.text = status
        markers.markers.append(text_marker)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleAvoidance()
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
