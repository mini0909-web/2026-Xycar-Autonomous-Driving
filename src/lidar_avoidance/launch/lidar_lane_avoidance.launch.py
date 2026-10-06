import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    lane_config = os.path.join(
        get_package_share_directory('lane_drive'), 'config', 'lane_drive.yaml'
    )
    avoidance_config = os.path.join(
        get_package_share_directory('lidar_avoidance'),
        'config',
        'lidar_avoidance.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/image_raw'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('auto_start', default_value='true'),
        DeclareLaunchArgument('avoidance_enabled', default_value='true'),
        Node(
            package='lane_drive',
            executable='lane_drive_node',
            name='lane_drive',
            output='screen',
            parameters=[
                lane_config,
                {
                    'image_topic': LaunchConfiguration('image_topic'),
                    # Do not publish directly to the VESC bridge in this launch.
                    'motor_topic': '/lane/drive_cmd',
                    'auto_start': ParameterValue(
                        LaunchConfiguration('auto_start'), value_type=bool
                    ),
                },
            ],
        ),
        Node(
            package='lidar_avoidance',
            executable='lidar_avoidance_node',
            name='lidar_avoidance',
            output='screen',
            parameters=[
                avoidance_config,
                {
                    'scan_topic': LaunchConfiguration('scan_topic'),
                    'enabled': ParameterValue(
                        LaunchConfiguration('avoidance_enabled'), value_type=bool
                    ),
                },
            ],
        ),
    ])
