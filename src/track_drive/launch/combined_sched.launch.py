# combined_sched.launch.py - 대회용: E2E 부스트 차선주행 + 라바콘 회피 + 동적장애물 회피 통합 런치
#   e2e_pure       → /motor_lane ──────────────────────────┐
#   cone_avoid     → /cmd/cone,/cone/active ────────────────┼─ [drive_mode_mux_sched] → /xycar_motor
#   dynamic_avoid  → /cmd/dynamic,/dynamic/active ──────────┘
#   combined.launch(연습용, 단일 속도 Q/E 고정)와 골격은 동일하고, mux만
#   drive_mux -> drive_mux_sched로 교체됨 (조향 작으면 프레임당1씩 부스트,
#   커지면 BASE_SPEED로 복귀, S/R로 정지/재개, 가속은 항상 프레임당1).
#   우선순위: traffic_light > cone > dynamic > lane
#   ros2 launch track_drive combined_sched.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def _xycar_cam():
    # xycar_device/xycar_cam 런치 재사용 (auto_white_balance/autoexposure 꺼진 usb_cam 설정)
    return GroupAction([
        SetRemap('image_raw', '/usb_cam/image_raw/front'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('xycar_cam'),
                             'launch', 'xycar_cam.launch.py')
            )
        ),
    ])


def generate_launch_description():
    start_motor = LaunchConfiguration('start_motor')
    use_traffic_light = LaunchConfiguration('use_traffic_light')
    ros1_bridge_setup = LaunchConfiguration('ros1_bridge_setup')

    return LaunchDescription([
        DeclareLaunchArgument('start_motor', default_value='true'),
        DeclareLaunchArgument('use_traffic_light', default_value='true'),
        DeclareLaunchArgument(
            'ros1_bridge_setup',
            default_value='/home/user/ros-humble-ros1-bridge/install/local_setup.bash',
        ),

        _xycar_cam(),

        # 라이다 → /scan
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('xycar_lidar'),
                             'launch', 'xycar_lidar.launch.py')
            )
        ),

        # ROS 1 VESC 모터 드라이버. 호스트의 USB 장치와 noetic_ws를 사용한다.
        ExecuteProcess(
            condition=IfCondition(start_motor),
            cmd=[
                'docker', 'run', '--rm', '--privileged',
                '--name', 'combined_sched_drive_ros1',
                '--network', 'host',
                '-e', 'ROS_MASTER_URI=http://localhost:11311',
                '-e', 'ROS_IP=127.0.0.1',
                '-v', '/home/user/noetic_ws:/root/noetic_ws',
                '-v', '/dev:/dev',
                '--group-add', 'dialout',
                'osrf/ros:noetic-xycar',
                'bash', '-c',
                (
                    'source /opt/ros/noetic/setup.bash; '
                    'source /root/noetic_ws/devel/setup.bash; '
                    'exec roslaunch xycar_motor xycar_motor.launch'
                ),
            ],
            output='screen',
        ),

        # roscore와 모터 노드가 준비된 뒤 ROS 2 /xycar_motor를 ROS 1로 전달한다.
        TimerAction(
            period=4.0,
            condition=IfCondition(start_motor),
            actions=[
                ExecuteProcess(
                    cmd=[
                        'bash', '-lc', [
                            'source /opt/ros/humble/setup.bash && source ',
                            ros1_bridge_setup,
                            ' && export ROS_MASTER_URI=http://localhost:11311',
                            ' && exec ros2 run ros1_bridge dynamic_bridge',
                            ' --bridge-all-topics',
                        ],
                    ],
                    output='screen',
                ),
            ],
        ),

        # E2E 차선 추론 → /motor_lane (부스트 판단은 drive_mux_sched가 함, e2e_pure는 원본 그대로)
        Node(package='track_drive', executable='e2e_pure',
             name='e2e_pure', output='screen'),

        # 라바콘 회피 → /cmd/cone, /cone/active
        Node(package='track_drive', executable='cone_avoid',
             name='cone_avoid', output='screen'),

        # 라이다 + 차선 인지 기반 동적 회피 → /cmd/dynamic, /dynamic/active
        Node(package='dynamic_obstacle_avoidance', executable='dynamic_obstacle_avoidance',
             name='dynamic_obstacle_avoidance', output='screen'),

        # 신호등: traffic_light_detector 패키지 하나로 통합
        # (YOLO박스+색상/슬롯 판정 + mux용 상태머신까지 노드 하나에서 처리).
        # use_traffic_light:=false로 끌 수 있음.
        Node(package='traffic_light_detector', executable='traffic_light_detector_node',
             name='traffic_light_detector', output='screen',
             parameters=[os.path.join(
                 get_package_share_directory('traffic_light_detector'),
                 'config', 'traffic_light.yaml')],
             condition=IfCondition(use_traffic_light)),

        # MUX(대회용): traffic_light > cone > dynamic > E2E lane, lane은 조향 작으면
        # 부스트/크면 BASE_SPEED 복귀 (가속 프레임당1, 감속 즉시), S/R로 정지/재개.
        Node(package='track_drive', executable='drive_mux_sched',
             name='drive_mode_mux_sched', output='screen'),
    ])
