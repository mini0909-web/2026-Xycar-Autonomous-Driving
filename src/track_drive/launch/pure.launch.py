# pure.launch.py - USB 카메라 + 순수 E2E 차선 주행 (신호등/게이트 없음)
#   ros2 launch track_drive pure.launch.py
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
        # 디버그 E2E: S/R/Q/E 키 제어, /xycar_motor 직접 발행
        Node(package='track_drive', executable='e2e_debug',
             name='e2e_debug', output='screen'),
    ])
