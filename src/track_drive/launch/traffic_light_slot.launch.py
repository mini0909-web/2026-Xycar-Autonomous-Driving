# traffic_light_slot.launch.py - 카메라 + traffic_light_slot(신호등 슬롯 판정) 단독 실행
#   ros2 launch track_drive traffic_light_slot.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap


def _xycar_cam():
    # xycar_cam의 네이티브 /image_raw를 traffic_light_slot이 구독하는
    # /usb_cam/image_raw/front로 remap (combined.launch.py와 동일 패턴)
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
    return LaunchDescription([
        _xycar_cam(),

        Node(package='track_drive', executable='traffic_light_slot',
             name='traffic_light_slot', output='screen'),
    ])
