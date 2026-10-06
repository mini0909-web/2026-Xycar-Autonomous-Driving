"""Launch the LiDAR, motor driver/ROS 1 bridge, and static avoidance node."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_lidar = LaunchConfiguration('start_lidar')
    start_motor = LaunchConfiguration('start_motor')
    lidar_params_file = LaunchConfiguration('lidar_params_file')
    noetic_workspace = LaunchConfiguration('noetic_workspace')
    ros1_bridge_setup = LaunchConfiguration('ros1_bridge_setup')

    lidar_share = FindPackageShare('xycar_lidar')
    lidar_launch = PathJoinSubstitution(
        [lidar_share, 'launch', 'xycar_lidar.launch.py']
    )
    default_lidar_params = PathJoinSubstitution(
        [lidar_share, 'params', 'ydlidar.yaml']
    )

    return LaunchDescription([
        DeclareLaunchArgument('start_lidar', default_value='true'),
        DeclareLaunchArgument('start_motor', default_value='true'),
        DeclareLaunchArgument(
            'lidar_params_file',
            default_value=default_lidar_params,
        ),
        DeclareLaunchArgument(
            'noetic_workspace',
            default_value='/home/user/noetic_ws',
        ),
        DeclareLaunchArgument(
            'ros1_bridge_setup',
            default_value=(
                '/home/user/ros-humble-ros1-bridge/install/local_setup.bash'
            ),
        ),
        DeclareLaunchArgument('straight_speed', default_value='5.0'),
        DeclareLaunchArgument('avoid_speed', default_value='5.0'),
        DeclareLaunchArgument('steering_angle', default_value='30.0'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lidar_launch),
            condition=IfCondition(start_lidar),
            launch_arguments={'params_file': lidar_params_file}.items(),
        ),

        # Run the ROS 1 VESC driver in the same host-networked container used
        # by the existing Xycar setup. Keeping docker in the foreground lets
        # ROS 2 launch stop and remove the container cleanly on Ctrl-C.
        ExecuteProcess(
            condition=IfCondition(start_motor),
            cmd=[
                'docker', 'run', '--rm',
                '--privileged',
                '--name', 'static_avoidance_ros1',
                '--network', 'host',
                '-e', 'ROS_MASTER_URI=http://localhost:11311',
                '-e', 'ROS_IP=127.0.0.1',
                '-v', [noetic_workspace, ':/root/noetic_ws'],
                '-v', '/dev:/dev',
                '--group-add', 'dialout',
                'osrf/ros:noetic-xycar',
                'bash', '-ci',
                (
                    'source /opt/ros/noetic/setup.bash; '
                    'source /root/noetic_ws/devel/setup.bash; '
                    'exec roslaunch xycar_motor xycar_motor.launch'
                ),
            ],
            output='screen',
        ),

        # Give roscore and the ROS 1 motor node time to start before bridging
        # /xycar_motor from ROS 2.
        TimerAction(
            period=4.0,
            condition=IfCondition(start_motor),
            actions=[
                ExecuteProcess(
                    cmd=[
                        'bash', '-lc',
                        [
                            'source /opt/ros/humble/setup.bash && source ',
                            ros1_bridge_setup,
                            ' && export ROS_MASTER_URI=http://localhost:11311',
                            ' && exec ros2 run ros1_bridge dynamic_bridge',
                            ' --bridge-all-topics',
                        ],
                    ],
                    output='screen',
                ),
            ],
        ),

        Node(
            package='static_obstacle_avoidance',
            executable='static_avoidance',
            name='static_obstacle_avoidance',
            output='screen',
            parameters=[{
                'straight_speed': ParameterValue(
                    LaunchConfiguration('straight_speed'), value_type=float
                ),
                'avoid_speed': ParameterValue(
                    LaunchConfiguration('avoid_speed'), value_type=float
                ),
                'steering_angle': ParameterValue(
                    LaunchConfiguration('steering_angle'), value_type=float
                ),
                # 0: automatically steer away from the obstacle side.
                'forced_direction': 0,
                'steering_sign': 1.0,
                'lateral_clearance': 0.30,
                'maximum_shift_time': 2.0,
                'pass_duration': 1.2,
                'scan_half_angle_deg': 120.0,
                'scan_max_distance': 1.0,
            }],
        ),
    ])
