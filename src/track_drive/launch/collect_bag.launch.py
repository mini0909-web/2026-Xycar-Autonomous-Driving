# collect_bag.launch.py - teleop_key로 수동조종하면서 카메라만 rosbag로 기록 (라이다 비활성화)
#   ros2 launch track_drive collect_bag.launch.py
import os
import datetime
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetRemap

BAG_ROOT = '/home/user/ruby_ws/bag'
BAG_TOPICS = ['/usb_cam/image_raw/front']  # 카메라만 기록 (라이다 비활성화)


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
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_out = os.path.join(BAG_ROOT, ts)
    return LaunchDescription([
        _xycar_cam(),

        Node(package='track_drive', executable='teleop_key',
             name='teleop_key', output='screen'),

        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', bag_out, *BAG_TOPICS],
            output='screen'),
    ])
