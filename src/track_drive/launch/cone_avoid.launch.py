# cone_avoid.launch.py - 라이다 + cone_avoid 단독 테스트 (mux 없음, 모터 도커는 별도로 켤 것)
#   ros2 launch track_drive cone_avoid.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # xycar_device/xycar_lidar 런치 재사용 (/scan 발행)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('xycar_lidar'),
                             'launch', 'xycar_lidar.launch.py')
            )
        ),
        Node(package='track_drive', executable='cone_avoid',
             name='cone_avoid', output='screen'),
    ])
