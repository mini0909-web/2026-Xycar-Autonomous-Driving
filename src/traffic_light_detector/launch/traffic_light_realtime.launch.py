"""Start the Xycar front camera and traffic-light detector together."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    detector_share = get_package_share_directory("traffic_light_detector")
    camera_share = get_package_share_directory("xycar_cam")

    default_params = os.path.join(detector_share, "config", "traffic_light.yaml")
    camera_launch = os.path.join(camera_share, "launch", "xycar_cam.launch.py")

    params_file = LaunchConfiguration("params_file")
    camera_start_delay = LaunchConfiguration("camera_start_delay")
    video_device = LaunchConfiguration("video_device")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Traffic-light detector parameter YAML",
        ),
        DeclareLaunchArgument(
            "camera_start_delay",
            default_value="2.0",
            description="Seconds to wait for xycar_cam before starting detection",
        ),
        DeclareLaunchArgument(
            "video_device",
            # Pass the resolved device path.  usb_cam on this Xycar turns the
            # relative target of the /dev/v4l/by-id symlink into the invalid
            # path /dev/../../video0 when the symlink itself is passed here.
            default_value="/dev/video0",
            description="Front camera V4L2 device",
        ),
        # Preserve the Xycar camera's native /image_raw and /camera_info topic
        # names.  The detector configuration subscribes to /image_raw.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(camera_launch),
            launch_arguments={"video_device": video_device}.items(),
        ),
        TimerAction(
            period=camera_start_delay,
            actions=[
                Node(
                    package="traffic_light_detector",
                    executable="traffic_light_detector_node",
                    name="traffic_light_detector",
                    output="screen",
                    # This launch starts xycar_cam without SetRemap, so the
                    # native topic is /image_raw here - override whatever the
                    # shared yaml's image_topic says (it's set for the
                    # SetRemap'd /usb_cam/image_raw/front used elsewhere).
                    parameters=[params_file, {"image_topic": "/image_raw"}],
                ),
            ],
        ),
    ])
