import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
    return LaunchDescription([
        _xycar_cam(),
        Node(package='track_drive', executable='teleop_key',
             name='teleop_key', output='screen'),
        Node(package='track_drive', executable='data_logger',
             name='data_logger', output='screen'),
    ])
