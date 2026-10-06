#!/usr/bin/env python3
"""ROS 2 arrow-key teleoperation node for Xycar."""

import os
import select
import sys
import termios
import time
import tty
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float32MultiArray

from .controller import (
    ControlConfig,
    KeyboardController,
    consume_key_buffer,
)


HELP = """
Arrow-key Xycar control
  UP/DOWN    : increase/decrease speed
  LEFT/RIGHT : steer left/right
  C          : center steering
  SPACE      : immediate stop and center
  Q          : stop and quit
Hold or repeatedly press a key; the deadman stops on lost input.
"""


class KeyboardDriveNode(Node):
    """Read the controlling terminal and publish [angle, speed]."""

    def __init__(self):
        super().__init__('keyboard_drive')
        self._declare_parameters()
        self.controller = KeyboardController(self._controller_config())
        self.motor_pub = self.create_publisher(
            Float32MultiArray, str(self._p('motor_topic')), 10
        )

        self.tty_fd: Optional[int] = None
        self.saved_terminal = None
        self.key_buffer = ''
        self.exit_requested = False
        self.exit_requested_at = 0.0
        self._open_terminal()

        rate = max(1.0, float(self._p('publish_rate_hz')))
        self.timer = self.create_timer(1.0 / rate, self._timer_callback)
        self.get_logger().info(
            f'KeyboardDrive ready: motor={self._p("motor_topic")}, '
            f'rate={rate:.1f}Hz'
        )
        sys.stdout.write(HELP)
        sys.stdout.flush()
        self._publish()

    def _declare_parameters(self):
        defaults: Dict[str, object] = {
            'motor_topic': '/xycar_motor',
            'speed_step': 1.0,
            'steering_step': 5.0,
            'max_forward_speed': 6.0,
            'max_reverse_speed': 3.0,
            'max_steering': 50.0,
            'left_steering_sign': -1.0,
            'deadman_timeout_sec': 0.5,
            'publish_rate_hz': 20.0,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _p(self, name):
        return self.get_parameter(name).value

    def _controller_config(self):
        return ControlConfig(
            speed_step=float(self._p('speed_step')),
            steering_step=float(self._p('steering_step')),
            max_forward_speed=float(self._p('max_forward_speed')),
            max_reverse_speed=float(self._p('max_reverse_speed')),
            max_steering=float(self._p('max_steering')),
            left_steering_sign=float(self._p('left_steering_sign')),
            deadman_timeout_sec=float(self._p('deadman_timeout_sec')),
        )

    def _open_terminal(self):
        try:
            self.tty_fd = os.open('/dev/tty', os.O_RDWR | os.O_NONBLOCK)
            self.saved_terminal = termios.tcgetattr(self.tty_fd)
            tty.setcbreak(self.tty_fd)
        except (OSError, termios.error) as error:
            if self.tty_fd is not None:
                os.close(self.tty_fd)
                self.tty_fd = None
            raise RuntimeError(
                'KeyboardDrive needs an interactive terminal (/dev/tty)'
            ) from error

    def restore_terminal(self):
        if self.tty_fd is None:
            return
        if self.saved_terminal is not None:
            termios.tcsetattr(
                self.tty_fd, termios.TCSADRAIN, self.saved_terminal
            )
        os.close(self.tty_fd)
        self.tty_fd = None

    def _read_commands(self):
        if self.tty_fd is None:
            return []
        chunks = []
        while select.select([self.tty_fd], [], [], 0.0)[0]:
            try:
                data = os.read(self.tty_fd, 64)
            except BlockingIOError:
                break
            if not data:
                break
            chunks.append(data.decode(errors='ignore'))
        if chunks:
            self.key_buffer += ''.join(chunks)
        commands, self.key_buffer = consume_key_buffer(self.key_buffer)
        return commands

    def _publish(self):
        msg = Float32MultiArray()
        msg.data = [
            float(self.controller.angle),
            float(self.controller.speed),
        ]
        self.motor_pub.publish(msg)

    def _show_status(self, reason=''):
        suffix = f' {reason}' if reason else ''
        sys.stdout.write(
            '\rangle={:6.1f} speed={:5.1f}{}      '.format(
                self.controller.angle, self.controller.speed, suffix
            )
        )
        sys.stdout.flush()

    def _timer_callback(self):
        now = time.monotonic()
        for command in self._read_commands():
            if self.controller.handle_key(command, now):
                self.exit_requested = True
                self.exit_requested_at = now
            self._show_status()

        if self.controller.apply_deadman(now):
            self._show_status('[DEADMAN STOP]')

        self._publish()

        # Keep publishing zero commands briefly before closing DDS on Q.
        if self.exit_requested and now - self.exit_requested_at >= 0.15:
            sys.stdout.write('\n')
            sys.stdout.flush()
            rclpy.shutdown()

    def publish_stop(self):
        self.controller.angle = 0.0
        self.controller.speed = 0.0
        self._publish()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = KeyboardDriveNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.publish_stop()
                rclpy.spin_once(node, timeout_sec=0.05)
            node.restore_terminal()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
