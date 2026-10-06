#!/usr/bin/env python3
"""Log dynamic-obstacle avoidance events without affecting control."""

from dataclasses import dataclass
import math
import re
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray


WIDTH_TEXT_PATTERN = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*cm\s*$', re.IGNORECASE)


@dataclass(frozen=True)
class ObstacleInfo:
    width_cm: float
    x: float
    y: float

    @property
    def distance(self) -> float:
        return math.hypot(self.x, self.y)


class DynamicObstacleLog(Node):
    """Subscribe to avoidance telemetry and print concise event logs."""

    def __init__(self) -> None:
        super().__init__('dynamic_obstacle_log')
        self.declare_parameters('', [
            ('status_topic', '/dynamic_obstacle_avoidance/status'),
            ('marker_topic', '/dynamic_obstacle_avoidance/markers'),
            ('active_topic', '/dynamic/active'),
            ('command_topic', '/cmd/dynamic'),
            ('cone_active_topic', '/cone/active'),
        ])

        self.previous_mode: Optional[str] = None
        self.previous_active: Optional[bool] = None
        self.previous_target_matched: Optional[bool] = None
        self.previous_cone_active: Optional[bool] = None
        self.latest_status: Dict[str, str] = {}
        self.visible_target: Optional[ObstacleInfo] = None
        self.last_obstacle: Optional[ObstacleInfo] = None
        self.obstacle_present = False
        self.max_lost_count = 0
        self.command_logged = False
        self.cone_active = False

        self.status_subscription = self.create_subscription(
            String, self._p('status_topic'), self._status_callback, 10)
        self.marker_subscription = self.create_subscription(
            MarkerArray, self._p('marker_topic'), self._marker_callback, 10)
        self.active_subscription = self.create_subscription(
            Bool, self._p('active_topic'), self._active_callback, 10)
        self.command_subscription = self.create_subscription(
            Float32MultiArray, self._p('command_topic'), self._command_callback, 10)
        self.cone_subscription = self.create_subscription(
            Bool, self._p('cone_active_topic'), self._cone_active_callback, 10)

        self.get_logger().info('장애물 회피 로그 노드 시작')

    def _p(self, name: str):
        return self.get_parameter(name).value

    @staticmethod
    def _parse_status(text: str) -> Dict[str, str]:
        values = {}
        for token in text.split():
            if '=' not in token:
                continue
            key, value = token.split('=', 1)
            values[key] = value
        return values

    @staticmethod
    def _parse_bool(value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        return None

    @staticmethod
    def _parse_count(value: Optional[str]) -> int:
        try:
            return int(value) if value is not None else 0
        except ValueError:
            return 0

    @staticmethod
    def _mode_text(mode: str) -> str:
        return {
            'WAITING_FOR_LIDAR': '라이다 대기',
            'E2E_FOLLOW': 'E2E 주행',
            'AVOIDING': '장애물 회피',
        }.get(mode, mode)

    @staticmethod
    def _direction_text(direction: str) -> str:
        return {
            'LEFT': '왼쪽',
            'RIGHT': '오른쪽',
            'NONE': '없음',
        }.get(direction, direction)

    @staticmethod
    def _yes_no(value: Optional[bool]) -> str:
        if value is None:
            return '알 수 없음'
        return '예' if value else '아니오'

    @staticmethod
    def _is_target_marker(marker: Marker) -> bool:
        return (
            marker.ns == 'clusters'
            and marker.type == Marker.CUBE
            and marker.action == Marker.ADD
            and marker.color.r >= 0.9
            and marker.color.g <= 0.3
            and marker.color.b <= 0.3
        )

    def _status_callback(self, msg: String) -> None:
        status = self._parse_status(msg.data)
        mode = status.get('state')
        if mode is None:
            return

        previous_mode = self.previous_mode
        if previous_mode is None:
            self.get_logger().info(f'[모드] {self._mode_text(mode)}')
        elif mode != previous_mode:
            self.get_logger().info(
                f'[모드] {self._mode_text(previous_mode)} → {self._mode_text(mode)}')

        target_matched = self._parse_bool(status.get('target_matched'))
        lost_count = self._parse_count(status.get('lost'))
        self.latest_status = status
        if self.obstacle_present:
            self.max_lost_count = max(self.max_lost_count, lost_count)

        if (
            self.previous_target_matched is True
            and target_matched is False
            and (mode == 'AVOIDING' or previous_mode == 'AVOIDING')
        ):
            self.get_logger().info('[추적 실패] 장애물 일치 실패')

        if (
            self.obstacle_present
            and mode != 'AVOIDING'
            and status.get('target', 'NONE') == 'NONE'
        ):
            final_lost_count = self.max_lost_count
            if (
                previous_mode == 'AVOIDING'
                and self.previous_target_matched is False
                and not self.cone_active
            ):
                final_lost_count += 1
            self._log_obstacle_end(final_lost_count)

        self.previous_mode = mode
        self.previous_target_matched = target_matched
        self._log_visible_obstacle_if_new()

    def _marker_callback(self, msg: MarkerArray) -> None:
        clusters = {}
        widths = {}

        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                continue
            if self._is_target_marker(marker):
                clusters[marker.id] = marker
                continue
            if (
                marker.ns == 'cluster_widths'
                and marker.type == Marker.TEXT_VIEW_FACING
                and marker.action == Marker.ADD
            ):
                match = WIDTH_TEXT_PATTERN.match(marker.text)
                if match is not None:
                    widths[marker.id] = float(match.group(1))

        self.visible_target = None
        for marker_id, cluster in clusters.items():
            width_cm = widths.get(marker_id)
            if width_cm is None or width_cm >= 30.0:
                continue
            self.visible_target = ObstacleInfo(
                width_cm=width_cm,
                x=float(cluster.pose.position.x),
                y=float(cluster.pose.position.y),
            )
            break

        self._log_visible_obstacle_if_new()

    def _log_visible_obstacle_if_new(self) -> None:
        if self.visible_target is None or self.obstacle_present:
            return
        if self.latest_status.get('target', 'NONE') == 'NONE':
            return

        info = self.visible_target
        side = self.latest_status.get('target', 'LEFT' if info.y >= 0.0 else 'RIGHT')
        avoidance = self.latest_status.get('avoid', 'NONE')
        locked = self._parse_bool(self.latest_status.get('target_locked'))
        matched = self._parse_bool(self.latest_status.get('target_matched'))
        confirm_count = self._parse_count(self.latest_status.get('confirm'))
        self.get_logger().info(
            f'[장애물 감지] 폭 {info.width_cm:.1f}cm | '
            f'위치 x={info.x:.2f}m, y={info.y:.2f}m | '
            f'거리 {info.distance:.2f}m | '
            f'방향 {self._direction_text(side)} | '
            f'회피 {self._direction_text(avoidance)} | '
            f'잠금 {self._yes_no(locked)} | '
            f'추적 {self._yes_no(matched)} | 확인 {confirm_count}회')
        self.last_obstacle = info
        self.obstacle_present = True
        self.max_lost_count = 0

    def _log_obstacle_end(self, lost_count: int) -> None:
        if not self.obstacle_present:
            return
        if self.last_obstacle is not None:
            self.get_logger().info(
                f'[장애물 종료] 마지막 폭 {self.last_obstacle.width_cm:.1f}cm | '
                f'소실 {lost_count}회')
        else:
            self.get_logger().info(f'[장애물 종료] 소실 {lost_count}회')
        self.obstacle_present = False
        self.visible_target = None
        self.last_obstacle = None
        self.max_lost_count = 0

    def _active_callback(self, msg: Bool) -> None:
        active = bool(msg.data)
        if self.previous_active is None:
            if active:
                self.get_logger().info('[회피 상태] 활성화')
        elif active != self.previous_active:
            state_text = '활성화' if active else '비활성화'
            self.get_logger().info(f'[회피 상태] {state_text}')

        if active != self.previous_active:
            self.command_logged = False
        self.previous_active = active

    def _command_callback(self, msg: Float32MultiArray) -> None:
        if self.previous_active is not True or self.command_logged or len(msg.data) < 2:
            return
        steering, speed = float(msg.data[0]), float(msg.data[1])
        self.get_logger().info(
            f'[회피 명령] 조향 {steering:.1f}도 | 속도 {speed:.1f}')
        self.command_logged = True

    def _cone_active_callback(self, msg: Bool) -> None:
        active = bool(msg.data)
        self.cone_active = active
        if self.previous_cone_active is None:
            if active:
                self.get_logger().info('[라바콘 우선] 활성화')
        elif active != self.previous_cone_active:
            state_text = '활성화' if active else '해제'
            self.get_logger().info(f'[라바콘 우선] {state_text}')
        self.previous_cone_active = active


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleLog()
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
