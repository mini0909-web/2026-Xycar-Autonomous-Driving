#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 2 LiDAR based static-obstacle avoidance for Xycar.

The node is intentionally standalone:
STRAIGHT -> SHIFT_OUT -> ALIGN_OUT -> PASS_OBSTACLE
         -> SHIFT_BACK -> ALIGN_BACK -> COOLDOWN -> STRAIGHT
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class ObstacleCluster:
    points: List[Tuple[float, float]]

    @property
    def min_x(self) -> float:
        return min(p[0] for p in self.points)

    @property
    def max_x(self) -> float:
        return max(p[0] for p in self.points)

    @property
    def min_y(self) -> float:
        return min(p[1] for p in self.points)

    @property
    def max_y(self) -> float:
        return max(p[1] for p in self.points)

    @property
    def x(self) -> float:
        return (self.min_x + self.max_x) / 2.0

    @property
    def y(self) -> float:
        return (self.min_y + self.max_y) / 2.0

    @property
    def distance(self) -> float:
        return min(math.hypot(x, y) for x, y in self.points)

    @property
    def width(self) -> float:
        return self.max_y - self.min_y

    @property
    def length(self) -> float:
        return self.max_x - self.min_x


class StaticObstacleAvoidance(Node):
    def __init__(self) -> None:
        super().__init__('static_obstacle_avoidance')
        self.declare_parameters('', [
            ('scan_topic', '/scan'),
            ('motor_topic', '/xycar_motor'),
            ('marker_topic', '~/markers'),
            ('straight_speed', 8.0),
            ('avoid_speed', 6.0),
            ('steering_angle', 40.0),
            ('steering_sign', 1.0),
            # 0: automatically select, 1: left, -1: right
            ('forced_direction', 0),
            # LiDAR mounting correction. Use 180 if the sensor faces backward.
            ('lidar_yaw_offset_deg', 0.0),
            ('scan_half_angle_deg', 90.0),
            ('scan_max_distance', 1.0),
            ('scan_timeout', 0.5),
            # Clustering
            ('cluster_gap_m', 0.18),
            ('minimum_cluster_points', 3),
            ('minimum_cluster_width', 0.03),
            ('maximum_cluster_width', 1.20),
            # Collision corridor
            ('detection_distance', 1.20),
            ('vehicle_half_width', 0.22),
            ('safety_margin', 0.12),
            ('emergency_stop_distance', 0.22),
            # Direction-selection sectors
            ('side_sector_min_angle_deg', 15.0),
            ('side_sector_max_angle_deg', 75.0),
            ('side_clearance_distance', 1.50),
            ('direction_hysteresis_m', 0.10),
            # State-machine timing and side tracking
            ('lateral_clearance', 0.42),
            ('minimum_shift_time', 0.20),
            ('maximum_shift_time', 1.50),
            ('minimum_pass_time', 0.40),
            ('maximum_pass_time', 3.00),
            ('pass_fallback_duration', 1.20),
            ('passed_clear_scan_count', 5),
            ('cooldown_time', 2.0),
        ])

        lidar_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.motor_pub = self.create_publisher(
            Float32MultiArray, self.param('motor_topic'), 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, self.param('marker_topic'), 10)
        self.create_subscription(
            LaserScan, self.param('scan_topic'), self.scan_callback, lidar_qos)

        self.points: List[Tuple[float, float]] = []
        self.clusters: List[ObstacleCluster] = []
        self.target: Optional[ObstacleCluster] = None
        self.last_scan_time = None
        self.scan_frame = ''
        self.state = 'WAITING_FOR_LIDAR'
        self.state_start = self.get_clock().now()
        self.avoid_direction = 1.0
        self.shift_duration = 0.5
        self.side_obstacle_seen = False
        self.clear_scan_count = 0
        self.timer = self.create_timer(0.05, self.control_callback)
        self.get_logger().info('Improved static obstacle avoidance ready')

    def param(self, name: str):
        return self.get_parameter(name).value

    def elapsed(self) -> float:
        return (self.get_clock().now() - self.state_start).nanoseconds / 1e9

    def set_state(self, new_state: str) -> None:
        if self.state == new_state:
            return
        self.state = new_state
        self.state_start = self.get_clock().now()
        self.status_pub.publish(String(data=new_state))
        self.get_logger().info('State changed: %s' % new_state)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def scan_callback(self, scan: LaserScan) -> None:
        self.last_scan_time = self.get_clock().now()
        self.scan_frame = scan.header.frame_id
        yaw_offset = math.radians(self.param('lidar_yaw_offset_deg'))
        half_angle = math.radians(self.param('scan_half_angle_deg'))
        max_distance = self.param('scan_max_distance')
        points = []

        # LaserScan order is angular order, which is preserved for clustering.
        for index, distance in enumerate(scan.ranges):
            if not math.isfinite(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue
            if distance > max_distance:
                continue
            raw_angle = scan.angle_min + index * scan.angle_increment
            angle = self.normalize_angle(raw_angle + yaw_offset)
            if abs(angle) > half_angle:
                continue
            points.append((
                distance * math.cos(angle),
                distance * math.sin(angle),
            ))

        self.points = points
        self.clusters = self.cluster_points(points)
        self.target = self.find_target_cluster()
        self.publish_markers(scan.header.stamp)
        if self.state == 'WAITING_FOR_LIDAR':
            self.set_state('STRAIGHT')

    def cluster_points(
        self, points: Sequence[Tuple[float, float]]
    ) -> List[ObstacleCluster]:
        if not points:
            return []
        result: List[ObstacleCluster] = []
        current = [points[0]]
        gap = self.param('cluster_gap_m')

        for point in points[1:]:
            if math.dist(current[-1], point) <= gap:
                current.append(point)
            else:
                self.append_valid_cluster(result, current)
                current = [point]
        self.append_valid_cluster(result, current)
        return result

    def append_valid_cluster(
        self,
        result: List[ObstacleCluster],
        points: List[Tuple[float, float]],
    ) -> None:
        if len(points) < self.param('minimum_cluster_points'):
            return
        cluster = ObstacleCluster(points.copy())
        if not (
            self.param('minimum_cluster_width')
            <= cluster.width
            <= self.param('maximum_cluster_width')
        ):
            return
        result.append(cluster)

    def find_target_cluster(self) -> Optional[ObstacleCluster]:
        corridor = self.param('vehicle_half_width') + self.param('safety_margin')
        candidates = [
            cluster for cluster in self.clusters
            if 0.0 < cluster.min_x <= self.param('detection_distance')
            and cluster.min_y <= corridor
            and cluster.max_y >= -corridor
        ]
        return min(candidates, key=lambda c: c.distance) if candidates else None

    def obstacle_detected(self) -> bool:
        return self.target is not None

    def sector_clearance(self, left: bool) -> float:
        minimum = math.radians(self.param('side_sector_min_angle_deg'))
        maximum = math.radians(self.param('side_sector_max_angle_deg'))
        limit = self.param('side_clearance_distance')
        distances = []
        for x, y in self.points:
            if x <= 0.0:
                continue
            angle = math.atan2(y, x)
            inside = minimum <= angle <= maximum if left else -maximum <= angle <= -minimum
            if inside:
                distances.append(math.hypot(x, y))
        return min(distances, default=limit)

    def choose_avoidance_direction(self) -> float:
        forced = int(self.param('forced_direction'))
        if forced in (-1, 1):
            return float(forced)

        left = self.sector_clearance(left=True)
        right = self.sector_clearance(left=False)
        hysteresis = self.param('direction_hysteresis_m')
        self.get_logger().info(
            'Clearance: left=%.2f m, right=%.2f m' % (left, right))

        if left > right + hysteresis:
            return 1.0
        if right > left + hysteresis:
            return -1.0
        # Similar clearance: avoid toward the side opposite the obstacle centre.
        return -1.0 if self.target is not None and self.target.y > 0.0 else 1.0

    def side_clusters(self) -> List[ObstacleCluster]:
        # After moving left, the obstacle must appear on the right, and vice versa.
        result = []
        for cluster in self.clusters:
            relative_side = -self.avoid_direction * cluster.y
            if cluster.distance <= 1.8 and relative_side >= 0.18:
                result.append(cluster)
        return result

    def obstacle_is_at_side(self) -> bool:
        return any(
            -self.avoid_direction * cluster.y
            >= self.param('lateral_clearance')
            for cluster in self.side_clusters()
        )

    def obstacle_passed(self) -> bool:
        side = self.side_clusters()
        if side:
            self.side_obstacle_seen = True
            self.clear_scan_count = 0
            # A cluster behind the LiDAR means the rear of the obstacle was passed.
            return any(cluster.max_x < -0.10 for cluster in side)

        if self.side_obstacle_seen:
            self.clear_scan_count += 1
        return (
            self.side_obstacle_seen
            and self.clear_scan_count >= self.param('passed_clear_scan_count')
        )

    def nearest_front_distance(self) -> float:
        corridor = self.param('vehicle_half_width') + self.param('safety_margin')
        return min(
            (
                math.hypot(x, y)
                for x, y in self.points
                if x > 0.0 and abs(y) <= corridor
            ),
            default=math.inf,
        )

    def lidar_is_valid(self) -> bool:
        if self.last_scan_time is None:
            return False
        age = (self.get_clock().now() - self.last_scan_time).nanoseconds / 1e9
        return age <= self.param('scan_timeout')

    def steering_command(self, direction: float) -> float:
        return (
            self.param('steering_sign')
            * direction
            * self.param('steering_angle')
        )

    def control_callback(self) -> None:
        if not self.lidar_is_valid():
            self.publish_motor(0.0, 0.0)
            return

        # Independent final safety layer.
        if self.nearest_front_distance() <= self.param('emergency_stop_distance'):
            self.publish_motor(0.0, 0.0)
            return

        straight_speed = self.param('straight_speed')
        avoid_speed = self.param('avoid_speed')

        if self.state == 'STRAIGHT':
            self.publish_motor(0.0, straight_speed)
            if self.obstacle_detected():
                self.avoid_direction = self.choose_avoidance_direction()
                self.side_obstacle_seen = False
                self.clear_scan_count = 0
                self.set_state('SHIFT_OUT')

        elif self.state == 'SHIFT_OUT':
            self.publish_motor(
                self.steering_command(self.avoid_direction), avoid_speed)
            if (
                self.elapsed() >= self.param('minimum_shift_time')
                and self.obstacle_is_at_side()
            ):
                self.shift_duration = self.elapsed()
                self.set_state('ALIGN_OUT')
            elif self.elapsed() >= self.param('maximum_shift_time'):
                self.get_logger().error('Could not confirm lateral clearance')
                self.set_state('ERROR_STOP')

        elif self.state == 'ALIGN_OUT':
            self.publish_motor(
                self.steering_command(-self.avoid_direction), avoid_speed)
            if self.elapsed() >= self.shift_duration:
                self.set_state('PASS_OBSTACLE')

        elif self.state == 'PASS_OBSTACLE':
            self.publish_motor(0.0, avoid_speed)
            minimum_time = self.elapsed() >= self.param('minimum_pass_time')
            sensor_passed = minimum_time and self.obstacle_passed()
            fallback_passed = (
                self.elapsed() >= self.param('pass_fallback_duration')
                and not self.side_obstacle_seen
            )
            if sensor_passed or fallback_passed:
                self.set_state('SHIFT_BACK')
            elif self.elapsed() >= self.param('maximum_pass_time'):
                self.get_logger().error('Obstacle passage was not confirmed')
                self.set_state('ERROR_STOP')

        elif self.state == 'SHIFT_BACK':
            self.publish_motor(
                self.steering_command(-self.avoid_direction), avoid_speed)
            if self.elapsed() >= self.shift_duration:
                self.set_state('ALIGN_BACK')

        elif self.state == 'ALIGN_BACK':
            self.publish_motor(
                self.steering_command(self.avoid_direction), avoid_speed)
            if self.elapsed() >= self.shift_duration:
                self.set_state('COOLDOWN')

        elif self.state == 'COOLDOWN':
            self.publish_motor(0.0, straight_speed)
            if self.elapsed() >= self.param('cooldown_time'):
                self.set_state('STRAIGHT')

        else:
            self.publish_motor(0.0, 0.0)

    def publish_motor(self, angle: float, speed: float) -> None:
        self.motor_pub.publish(Float32MultiArray(
            data=[float(angle), float(speed)]))

    @staticmethod
    def marker_point(x: float, y: float) -> Point:
        return Point(x=float(x), y=float(y), z=0.0)

    def base_marker(
        self, marker_id: int, marker_type: int, stamp, namespace: str
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.scan_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.a = 0.85
        return marker

    def publish_markers(self, stamp) -> None:
        if not self.scan_frame:
            return
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        scan = self.base_marker(0, Marker.POINTS, stamp, 'scan_points')
        scan.scale.x = scan.scale.y = 0.025
        scan.color.g = 0.9
        scan.color.b = 0.2
        scan.points = [self.marker_point(x, y) for x, y in self.points]
        array.markers.append(scan)

        corridor = self.base_marker(0, Marker.CUBE, stamp, 'collision_corridor')
        corridor.pose.position.x = self.param('detection_distance') / 2.0
        corridor.scale.x = self.param('detection_distance')
        corridor.scale.y = 2.0 * (
            self.param('vehicle_half_width') + self.param('safety_margin'))
        corridor.scale.z = 0.01
        corridor.color.r = 1.0
        corridor.color.g = 0.75
        corridor.color.a = 0.18
        array.markers.append(corridor)

        for index, cluster in enumerate(self.clusters):
            box = self.base_marker(index, Marker.CUBE, stamp, 'cluster_boxes')
            box.pose.position.x = cluster.x
            box.pose.position.y = cluster.y
            box.pose.position.z = 0.10
            box.scale.x = max(cluster.length, 0.08)
            box.scale.y = max(cluster.width, 0.08)
            box.scale.z = 0.20
            if cluster is self.target:
                box.color.r, box.color.g, box.color.b = 1.0, 0.1, 0.1
            else:
                box.color.r, box.color.g, box.color.b = 0.1, 0.7, 1.0
            array.markers.append(box)

        arrow = self.base_marker(0, Marker.ARROW, stamp, 'avoid_direction')
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.04, 0.08, 0.10
        arrow.color.g, arrow.color.b = 1.0, 0.2
        arrow.points = [
            self.marker_point(0.0, 0.0),
            self.marker_point(0.8, 0.45 * self.avoid_direction),
        ]
        array.markers.append(arrow)

        text = self.base_marker(0, Marker.TEXT_VIEW_FACING, stamp, 'state')
        text.pose.position.x = 0.5
        text.pose.position.z = 0.45
        text.scale.z = 0.16
        text.color.r, text.color.g, text.color.b = 1.0, 1.0, 0.1
        text.text = self.state
        array.markers.append(text)
        self.marker_pub.publish(array)

    def destroy_node(self):
        self.publish_motor(0.0, 0.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StaticObstacleAvoidance()
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