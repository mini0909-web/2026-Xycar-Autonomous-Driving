"""Launch sensors, E2E lane driving, LiDAR avoidance, and the mux."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_camera = LaunchConfiguration('start_camera')
    start_lidar = LaunchConfiguration('start_lidar')
    start_motor = LaunchConfiguration('start_motor')

    lidar_launch = PathJoinSubstitution([
        FindPackageShare('xycar_lidar'), 'launch', 'xycar_lidar.launch.py'
    ])
    camera_launch = PathJoinSubstitution([
        FindPackageShare('xycar_cam'), 'launch', 'xycar_cam.launch.py'
    ])
    lane_drive_launch = PathJoinSubstitution([
        FindPackageShare('lane_drive'), 'launch', 'lane_drive.launch.py'
    ])

    return LaunchDescription([
        # Tuning is intentionally kept in the Python nodes/config YAML, not in
        # launch arguments.  These switches only control which hardware starts.
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_lidar', default_value='true'),
        DeclareLaunchArgument('start_motor', default_value='true'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(camera_launch),
            condition=IfCondition(start_camera),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lidar_launch),
            condition=IfCondition(start_lidar),
        ),

        ExecuteProcess(
            condition=IfCondition(start_motor),
            cmd=[
                'bash', '-lc',
                'docker rm -f ros1_container >/dev/null 2>&1 || true; '
                'exec docker run --rm -d --privileged --name ros1_container '
                '--network host -e DISPLAY -e ROS_MASTER_URI=http://localhost:11311 '
                '-e ROS_IP=127.0.0.1 -v /tmp/.X11-unix:/tmp/.X11-unix '
                '-v /home/user/Downloads:/root/Downloads '
                '-v /home/user/noetic_ws:/root/noetic_ws -v /dev:/dev '
                '--group-add dialout osrf/ros:noetic-xycar '
                'bash -ci "source /opt/ros/noetic/setup.bash; '
                'source /root/noetic_ws/devel/setup.bash; '
                'export ROS_MASTER_URI=http://localhost:11311; '
                'unset ROS_HOSTNAME; export ROS_IP=127.0.0.1; '
                'exec roslaunch xycar_motor xycar_motor.launch"',
            ],
            output='screen',
        ),
        TimerAction(
            period=4.0,
            condition=IfCondition(start_motor),
            actions=[ExecuteProcess(
                cmd=[
                    'bash', '-lc',
                    'source /opt/ros/humble/setup.bash && '
                    'source /home/user/ros-humble-ros1-bridge/install/local_setup.bash && '
                    'export ROS_MASTER_URI=http://localhost:11311 && '
                    'exec ros2 run ros1_bridge dynamic_bridge --bridge-all-topics',
                ],
                output='screen',
            )],
        ),
        RegisterEventHandler(OnShutdown(on_shutdown=[
            ExecuteProcess(
                condition=IfCondition(start_motor),
                cmd=['docker', 'stop', 'ros1_container'], output='screen',
            ),
        ])),

        # lane_drive remains the mux's E2E fallback. Its lane-perception topics
        # are no longer inputs or preconditions for dynamic avoidance.
        IncludeLaunchDescription(PythonLaunchDescriptionSource(lane_drive_launch)),
        Node(
            package='dynamic_obstacle_avoidance',
            executable='dynamic_obstacle_avoidance',
            name='dynamic_obstacle_avoidance', output='screen',
        ),
        Node(package='track_drive', executable='drive_mux', name='drive_mode_mux', output='screen'),
    ])
