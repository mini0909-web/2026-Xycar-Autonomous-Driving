import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("traffic_light_detector")
    default_config = os.path.join(share, "config", "traffic_light.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_config),
        Node(
            package="traffic_light_detector",
            executable="traffic_light_detector_node",
            name="traffic_light_detector",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
