# e2e.launch.py - 단일카메라 E2E 주행 + 신호등 출발 게이트
#   ros2 launch track_drive e2e.launch.py
#   (먼저: endpoint 실행 -> sim Play -> 카메라 hz 로 깨우기 -> 이 launch)
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # 신호등 규칙노드: /state/traffic_go (초록=출발허가) 발행
        Node(package="track_drive", executable="traffic_light",
             name="traffic_light", output="screen"),
        # E2E 추론 주행: 출발 전 정지, 초록 보면 출발 후 차선주행
        Node(package="track_drive", executable="e2e_drive",
             name="e2e_drive", output="screen"),
    ])
