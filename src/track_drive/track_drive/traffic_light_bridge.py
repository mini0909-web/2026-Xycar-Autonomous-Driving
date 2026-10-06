#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# traffic_light_bridge.py - traffic_light_detector 패키지(/traffic_light/state)를
#   구독해서 traffic_light_yolo.py/traffic_light_slot.py와 동일한 상태머신을 돌리고
#   /cmd/traffic_light, /traffic_light/active로 발행해 mux에 연동한다.
#   RED/YELLOW/UNKNOWN은 무시(둘 다 안 보이는 것으로 취급).
#     GREEN(=slot3, 직진) -> STRAIGHT: 1회성 출발 게이트
#     LEFT(=slot2)        -> STRAIGHT_LEFT: 사라지는 순간 좌회전 발동(재발동 가능)
#   traffic_light_detector_node 자체의 debounce(temporal_filter.confirm_frames)와는
#   별개로, 여기서도 CONFIRM_FRAMES 디바운스를 한 번 더 걸어 이중으로 오탐을 줄인다.
#
#   디버그: traffic_light_detector_node가 발행하는 /traffic_light/debug/compressed를
#   구독해서 토픽으로만 보는 대신 cv2 창으로 직접 띄우고, 그 위에 이 노드의 상태머신
#   상태(게이트/좌회전 active)까지 같이 그려서 표시한다.

import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Bool, Float32MultiArray

LEFT_ANGLE = -30.0
LEFT_SPEED = 20.0
LEFT_DURATION_SEC = 1.0
CONFIRM_FRAMES = 3  # 이 프레임 연속 검출돼야 진짜로 인정 (오탐 방지 디바운스)
LEFT_ABSENT_FRAMES = 3  # 확정된 LEFT가 이만큼 연속 미검출돼야 "진짜 사라짐"으로 인정
RED_BLOCK_FRAMES = 5  # 최근 이만큼 프레임 안에 RED가 보였으면 GREEN이 잡혀도
                      # 게이트를 열지 않음 (실제 신호는 빨강/초록이 동시에 못 켜지니,
                      # 번갈아 잡히는 것 자체가 지금 판정이 불안정하다는 신호)

WIN = 'TL-Bridge'
DISPLAY_SCALE = 2.4  # 디버그 창을 이 배율로 키워서 표시


class TrafficLightBridge(Node):

    def __init__(self):
        super().__init__('traffic_light_bridge')

        self.gate_opened = False
        self.frames_since_red = 10 ** 9
        self.straight_streak = 0
        self.left_streak = 0
        self.left_confirmed_visible = False
        self.left_absent_streak = 0
        self.left_until = None
        self.last_state = 'NONE'
        self.last_active = False
        self.last_angle = 0.0
        self.last_speed = 0.0
        self.vis = None

        self.pub_cmd = self.create_publisher(Float32MultiArray, '/cmd/traffic_light', 10)
        self.pub_active = self.create_publisher(Bool, '/traffic_light/active', 10)
        self.create_subscription(String, '/traffic_light/state', self.cb, 10)
        self.create_subscription(CompressedImage, '/traffic_light/debug/compressed',
                                 self.cb_debug_image, 10)

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        self.create_timer(0.1, self._render_timer)
        self.get_logger().info('traffic_light_bridge ready | listening /traffic_light/state')

    def _update_state_machine(self, straight_visible, left_visible):
        now = time.monotonic()

        self.straight_streak = self.straight_streak + 1 if straight_visible else 0
        straight_confirmed = self.straight_streak >= CONFIRM_FRAMES
        red_recent = self.frames_since_red < RED_BLOCK_FRAMES
        if not self.gate_opened and straight_confirmed:
            if red_recent:
                self.get_logger().warn(
                    'GREEN 확정됐지만 최근 {}프레임 안에 RED도 보여서 불안정 판정으로 보고 게이트 보류'
                    .format(RED_BLOCK_FRAMES))
            else:
                self.gate_opened = True
                self.get_logger().info(
                    'GREEN(STRAIGHT) {}프레임 연속 확인 -> 출발 게이트 해제(1회성)'.format(CONFIRM_FRAMES))

        # LEFT 등장 디바운스: CONFIRM_FRAMES 연속 보여야 "확정 감지"로 인정.
        self.left_streak = self.left_streak + 1 if left_visible else 0
        if self.left_streak >= CONFIRM_FRAMES:
            self.left_confirmed_visible = True
            self.left_absent_streak = 0
        elif self.left_confirmed_visible:
            # 확정된 상태에서 미검출 프레임이 이어지는 중 - 사라짐도 디바운스한다.
            # 잠깐 한두 프레임 놓쳐도(인지가 순간 끊겨도) 바로 좌회전이 발동하지
            # 않도록, LEFT_ABSENT_FRAMES 연속으로 안 보여야 "진짜 사라짐"으로 본다.
            if left_visible:
                self.left_absent_streak = 0
            else:
                self.left_absent_streak += 1
                if self.left_absent_streak >= LEFT_ABSENT_FRAMES:
                    self.left_until = now + LEFT_DURATION_SEC
                    self.get_logger().info(
                        'LEFT {}프레임 연속 미검출 -> 좌회전 발동 (재발동 가능)'.format(LEFT_ABSENT_FRAMES))
                    self.left_confirmed_visible = False
                    self.left_absent_streak = 0

        if self.left_until is not None and now < self.left_until:
            return True, LEFT_ANGLE, LEFT_SPEED
        self.left_until = None

        if not self.gate_opened:
            return True, 0.0, 0.0

        return False, 0.0, 0.0

    def cb(self, msg):
        state = msg.data
        self.last_state = state
        straight_visible = state == 'GREEN'
        left_visible = state == 'LEFT'
        # RED/YELLOW/UNKNOWN은 게이트/좌회전 판단엔 안 쓰지만, RED는 최근 등장 여부를
        # 별도로 추적해서 GREEN 오탐 방지용 가드에 쓴다.
        self.frames_since_red = 0 if state == 'RED' else self.frames_since_red + 1

        active, angle, speed = self._update_state_machine(straight_visible, left_visible)
        self.last_active, self.last_angle, self.last_speed = active, angle, speed
        self.pub_active.publish(Bool(data=active))
        if active:
            self.pub_cmd.publish(Float32MultiArray(data=[angle, speed]))

    def cb_debug_image(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        dbg = cv2.resize(img, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                          interpolation=cv2.INTER_LINEAR)
        dh = dbg.shape[0]
        cv2.putText(dbg, 'state:{}'.format(self.last_state),
                    (10, dh - 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        cv2.putText(dbg, 'gate:{}  active:{} ({:+.0f},{:.0f})  frames_since_red:{}'.format(
                    int(self.gate_opened), int(self.last_active),
                    self.last_angle, self.last_speed,
                    min(self.frames_since_red, 999)),
                    (10, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        self.vis = dbg

    def _render_timer(self):
        if self.vis is not None:
            cv2.imshow(WIN, self.vis)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
