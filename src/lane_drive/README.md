# lane_drive

ROS 2 node that detects white/yellow lanes from `/image_raw`, calculates a
filtered PD steering command, and publishes `/xycar_motor` as
`std_msgs/msg/Float32MultiArray` containing `[angle, speed]`.

The input image is rectified with `config/fisheye_camera.yaml` before lane
detection. The rectified camera image is published on `/lane/rectified_image`.

## Run safely

Start the camera first, then launch the node. The default `auto_start` value is
false, so the node will detect lanes and publish debug images without moving.

```bash
source /opt/ros/humble/setup.bash
source ~/xycar_ws/install/setup.bash
source ~/ruby_ws/install/setup.bash

ros2 launch xycar_cam xycar_cam.launch.py
ros2 launch lane_drive lane_drive.launch.py
```

Inspect the detector before enabling the motor:

```bash
ros2 run rqt_image_view rqt_image_view /lane/debug_image
ros2 topic echo /lane/error
```

Enable lane driving with the existing `DriveEnable` LKAS_MAIN mode:

```bash
ros2 service call /drive_enable drive_control_msgs/srv/DriveEnable "{mode: 4}"
```

Any other mode disables this node and sends a stop command:

```bash
ros2 service call /drive_enable drive_control_msgs/srv/DriveEnable "{mode: 0}"
```

For a first bench test, keep the wheels off the ground and reduce `max_speed`
and `min_speed` in `config/lane_drive.yaml`. If the wheels steer away from the
lane center, change `steering_sign` from `1.0` to `-1.0`.

## Topics and service

- Subscribe `/image_raw`: `sensor_msgs/msg/Image`
- Publish `/lane/rectified_image`: `sensor_msgs/msg/Image`
- Publish `/xycar_motor`: `std_msgs/msg/Float32MultiArray`
- Publish `/lane/error`: `std_msgs/msg/Float32`
- Publish `/lane/yellow_side`: `std_msgs/msg/Int8` (`-1` left, `1` right,
  `0` unknown)
- Publish `/lane/debug_image`: `sensor_msgs/msg/Image`
- Publish `/lane/mask`: `sensor_msgs/msg/Image`
- Serve `/drive_enable`: `drive_control_msgs/srv/DriveEnable`

If camera frames stop or lanes remain lost, the node publishes `[0.0, 0.0]`.

## Interactive HSV tuner

The tuner subscribes to the camera, applies the same fisheye correction and
ROI as the lane detector, and never publishes a motor command.

```bash
ros2 launch lane_drive hsv_tuner.launch.py
```

Adjust the six HSV trackbars while watching the WHITE, YELLOW, and COMBINED
masks. Click `SAVE YAML` in the preview (or press `S`) to update the source and
installed `lane_drive.yaml`. Restart `lane_drive` after saving. Press `Q`, Esc,
or Ctrl-C to close the tuner.

`detection_bottom_ratio` keeps the calibrated image size unchanged and removes
only the lower part of the lane mask. A red `DETECTION BOTTOM` line is drawn in
both the tuner preview and `/lane/debug_image`.
