import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('keyboard_drive'),
        'config',
        'keyboard_drive.yaml',
    )

    return LaunchDescription([
        Node(
            package='keyboard_drive',
            executable='keyboard_drive_node',
            name='keyboard_drive',
            output='screen',
            emulate_tty=True,
            parameters=[config],
        ),
    ])
