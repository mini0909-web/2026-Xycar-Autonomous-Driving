"""Launch the interactive HSV tuner without any motor publisher."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _source_config_or_installed() -> Path:
    share_directory = Path(get_package_share_directory('lane_drive'))
    installed_config = share_directory / 'config' / 'lane_drive.yaml'
    for parent in share_directory.parents:
        source_config = parent / 'src' / 'lane_drive' / 'config' / 'lane_drive.yaml'
        if source_config.exists():
            return source_config
    return installed_config


def generate_launch_description():
    share_directory = Path(get_package_share_directory('lane_drive'))
    installed_config = share_directory / 'config' / 'lane_drive.yaml'
    source_config = _source_config_or_installed()

    return LaunchDescription([
        DeclareLaunchArgument(
            'image_topic',
            default_value='/image_raw',
            description='Camera image topic used for HSV tuning.',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=str(source_config),
            description='Source lane_drive.yaml modified by SAVE YAML.',
        ),
        DeclareLaunchArgument(
            'calibration_file',
            default_value='',
            description='Optional fisheye calibration YAML override.',
        ),
        Node(
            package='lane_drive',
            executable='hsv_tuner_node',
            name='hsv_tuner',
            output='screen',
            parameters=[{
                'image_topic': LaunchConfiguration('image_topic'),
                'config_file': LaunchConfiguration('config_file'),
                'runtime_config_file': str(installed_config),
                'calibration_file': LaunchConfiguration('calibration_file'),
            }],
        ),
    ])
