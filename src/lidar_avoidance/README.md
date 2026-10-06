# lidar_avoidance

This package places one safety/avoidance supervisor between `lane_drive` and
the final Xycar motor topic.

## Run

Start the camera and LiDAR drivers first, then use the combined launch file:

```bash
ros2 launch xycar_cam xycar_cam.launch.py
ros2 launch xycar_lidar xycar_lidar.launch.py
ros2 launch lidar_avoidance lidar_lane_avoidance.launch.py
```

Do not launch `lane_drive lane_drive.launch.py` at the same time. The combined
launch already starts it with its motor output remapped to `/lane/drive_cmd`.

## Topics

- Subscribes `/scan` (`sensor_msgs/LaserScan`).
- Subscribes `/lane/drive_cmd` (`std_msgs/Float32MultiArray`).
- Subscribes `/lane/yellow_side` (`std_msgs/Int8`): `-1` left, `1` right,
  and `0` unknown.
- Publishes `/xycar_motor` (`std_msgs/Float32MultiArray`) as the only final
  motor command publisher.
- Publishes `/lidar_avoidance/state` and `/lidar_avoidance/debug`.

The debug array is `[state, front_count, front_range_m, target_count,
target_range_m, steering_bias, yellow_side]`.

```bash
ros2 topic echo /lidar_avoidance/state
ros2 topic echo /lidar_avoidance/debug
ros2 topic echo /lane/yellow_side
```

An emergency stop is latched. After removing the obstacle, reset it with:

```bash
ros2 service call /lidar_avoidance/reset std_srvs/srv/Trigger "{}"
```

## Safety note

The default avoidance speed cap is intentionally 8.0. Confirm LiDAR
left/right orientation, detection rectangles, motor steering sign, and the
S-curve duration with the car lifted and then at low speed before increasing
the cap.
