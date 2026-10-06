# cluster_debug.launch.py - 라이다 + 클러스터 디버그 (모터 제어 없음, 순수 확인용)
#   ros2 launch track_drive cluster_debug.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('xycar_lidar'),
                             'launch', 'xycar_lidar.launch.py')
            )
        ),
        Node(package='track_drive', executable='cluster_debug',
             name='cluster_debug', output='screen'),
    ])
