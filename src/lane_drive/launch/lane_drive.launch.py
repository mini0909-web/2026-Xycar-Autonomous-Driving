import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('lane_drive'), 'config', 'lane_drive.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/image_raw',
            description='Camera image topic consumed by the lane detector.',
        ),
        DeclareLaunchArgument(
            'auto_start',
            default_value='true',
            description='Start driving immediately when valid lanes are detected.',
        ),
        DeclareLaunchArgument(
            'motor_topic',
            default_value='/control/lane',
            description='Lane command output topic consumed by drive_mux.',
        ),
        Node(
            package='lane_drive',
            executable='lane_drive_node',
            name='lane_drive',
            output='screen',
            parameters=[
                config,
                {
                    'image_topic': LaunchConfiguration('image_topic'),
                    'motor_topic': LaunchConfiguration('motor_topic'),
                    'auto_start': ParameterValue(
                        LaunchConfiguration('auto_start'), value_type=bool
                    ),
                },
            ],
        ),
    ])
