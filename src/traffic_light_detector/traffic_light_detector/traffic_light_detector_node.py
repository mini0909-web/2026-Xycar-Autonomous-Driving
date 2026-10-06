"""ROS 2 I/O wrapper for :mod:`traffic_light_detector.detector_core`.

mux 연동용 상태머신(원래 track_drive의 traffic_light_bridge.py 별도 노드였던 것)을
여기 직접 합쳤다 - 프로세스 하나 줄이고, 디버그 이미지도 압축발행/구독/디코드
왕복 없이 core.detect()가 만든 overlay를 바로 화면에 띄운다.
  RED/YELLOW/UNKNOWN은 게이트/좌회전 판단에 안 씀.
    GREEN(=slot3, 직진) -> STRAIGHT: 1회성 출발 게이트 (단, 최근 RED_BLOCK_FRAMES
      프레임 안에 RED도 보였으면 판정이 불안정하다고 보고 게이트 보류)
    LEFT(=slot2)        -> STRAIGHT_LEFT: LEFT_ABSENT_FRAMES 연속 미검출돼야
      "진짜 사라짐"으로 보고 좌회전 발동(재발동 가능)
"""

from __future__ import annotations

import time
from typing import Dict

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32MultiArray, String

from .detector_core import DEFAULT_CONFIG, STATES, TrafficLightDetectorCore


EXTRA_DEFAULTS = {
    "image_topic": "/usb_cam/image_raw/front",
    "input_compressed": False,
    "qos.reliability": "best_effort",
    "qos.depth": 5,
    "input_timeout_sec": 0.75,
    "debug.publish_images": False,
    "debug.jpeg_quality": 85,
    "gui.enabled": False,
    "gui.config_save_path": "",
    "gui.show_original": True,
    "gui.show_masks": True,
    "gui.show_overlay": True,
    "gui.max_processing_fps": 10.0,
}

LEFT_ANGLE = -30.0
LEFT_SPEED = 20.0
LEFT_DURATION_SEC = 1.0
STRAIGHTEN_DURATION_SEC = 0.1  # 좌회전 끝난 직후 조향을 0으로 정렬하는 시간
CONFIRM_FRAMES = 3  # 이 프레임 연속 검출돼야 진짜로 인정 (오탐 방지 디바운스)
LEFT_ABSENT_FRAMES = 12  # 확정된 LEFT가 이만큼 연속 미검출돼야 "진짜 사라짐"으로 인정
RED_BLOCK_FRAMES = 5  # 최근 이만큼 프레임 안에 RED가 보였으면 GREEN이 잡혀도
                      # 게이트를 열지 않음 (실제 신호는 빨강/초록이 동시에 못 켜지니,
                      # 번갈아 잡히는 것 자체가 지금 판정이 불안정하다는 신호)

WIN = "TL-Detector"
DISPLAY_SCALE = 2.4  # 디버그 창을 이 배율로 키워서 표시

# 정지 상태(출발 전)에서는 신호등 헤드 바로 위/뒤 형광등 오탐을 막으려고 좁은
# ROI(search_roi.y=0.20~0.55, yaml 기본값)를 쓰지만, 출발하고 나면(=게이트 1회성
# 해제) 접근하면서 원근 때문에 신호등이 화면 위쪽으로 올라가서 좁은 ROI 밖으로
# 벗어나버린다 - 이걸 "신호가 꺼짐"으로 오판해 좌회전이 조기 발동되거나 아예
# 놓치는 문제가 있었다. 그래서 게이트가 열리는 순간 딱 한 번 ROI를 원래 범위
# (y=0~0.55)로 넓혀서 주행 중엔 계속 이 값을 쓴다.
POST_GATE_ROI_Y = 0.0
POST_GATE_ROI_H = 0.55

# 혹시 GREEN 자동감지도 안 되고 t키도 못 눌렀을 상황을 대비한 안전장치 -
# 노드 시작 후 이만큼 지나면 게이트를 자동으로 강제 통과시킨다(1회성, 이미
# 열려있으면 아무 효과 없음).
GATE_TIMEOUT_SEC = 15.0


def declare_parameters(node: Node) -> Dict[str, object]:
    defaults = dict(DEFAULT_CONFIG); defaults.update(EXTRA_DEFAULTS)
    for name, value in defaults.items():
        if not node.has_parameter(name):
            node.declare_parameter(name, value)
    return {name: node.get_parameter(name).value for name in defaults}


def camera_qos(values: Dict[str, object]) -> QoSProfile:
    reliability = (ReliabilityPolicy.RELIABLE
                   if str(values.get("qos.reliability", "best_effort")).lower() == "reliable"
                   else ReliabilityPolicy.BEST_EFFORT)
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(values.get("qos.depth", 5))),
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


class TrafficLightDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("traffic_light_detector")
        self.values = declare_parameters(self)
        self.bridge = CvBridge(); self.core = TrafficLightDetectorCore(self.values)
        self.state_pub = self.create_publisher(String, "/traffic_light/state", 10)
        self.score_pub = self.create_publisher(Float32MultiArray, "/traffic_light/scores", 10)
        # 디버그 이미지는 더 이상 압축 발행하지 않는다 - traffic_light_bridge가
        # 이 노드에 통합돼서 토픽으로 구독하는 곳이 없어졌고, cv2 창으로 직접
        # 보여주면 되므로 JPEG 인코드+발행 비용을 아낀다.
        self.mask_pub = self.create_publisher(CompressedImage, "/traffic_light/mask/compressed", 2)

        # mux 연동 상태머신(구 traffic_light_bridge.py) 상태
        self.gate_opened = False
        self.frames_since_red = 10 ** 9
        self.straight_streak = 0
        self.left_streak = 0
        self.left_confirmed_visible = False
        self.left_absent_streak = 0
        self.left_until = None
        self.straighten_until = None
        self.last_active = False
        self.last_angle = 0.0
        self.last_speed = 0.0
        self.vis = None
        self._latest_overlay = None
        self._latest_state = 'NONE'
        self.cmd_pub = self.create_publisher(Float32MultiArray, "/cmd/traffic_light", 10)
        self.active_pub = self.create_publisher(Bool, "/traffic_light/active", 10)
        # MUX 창(t키)에서 강제 통과 요청이 오면 이 창을 따로 클릭 안 해도
        # 게이트가 열리게 - cv2.waitKey는 자기 창에 포커스 있을 때만 키를
        # 잡아서, 토픽으로 넘겨받아야 다른 창에서 누른 t도 반영된다.
        self.create_subscription(Bool, "/traffic_light/force_gate", self._on_force_gate, 10)

        topic = str(self.values["image_topic"])
        message_type = CompressedImage if bool(self.values["input_compressed"]) else Image
        self.subscription = self.create_subscription(message_type, topic, self._image_callback,
                                                     camera_qos(self.values))
        self.add_on_set_parameters_callback(self._parameters_changed)
        self.last_image_wall = time.monotonic(); self.stale_published = False
        self.timeout_timer = self.create_timer(0.20, self._check_input_timeout)

        # 안전장치: GREEN 자동감지/t키 둘 다 놓쳤을 경우를 대비해 노드 시작
        # GATE_TIMEOUT_SEC 후 자동으로 게이트를 강제 통과시킨다.
        self._node_start_time = time.monotonic()
        self.gate_timeout_timer = self.create_timer(0.5, self._check_gate_timeout)

        # 시각화 창 꺼둠(경량화) - namedWindow/resizeWindow/렌더타이머 자체를
        # 안 만들어서 매 프레임 리사이즈+putText 비용이 안 든다. 로컬 t키는
        # 이 창이 있어야 동작했는데, MUX 창에서 t 누르면 토픽으로 넘어오는
        # /traffic_light/force_gate 경로랑 GATE_TIMEOUT_SEC 안전장치가 있어서
        # 게이트 여는 방법 자체는 그대로 남아있다.
        # cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(WIN, int(640 * DISPLAY_SCALE), int(480 * DISPLAY_SCALE))
        # self.create_timer(0.1, self._render_timer)

        self.get_logger().info(
            f"subscribing to {topic} ({message_type.__name__}), QoS="
            f"{self.values['qos.reliability']}; mux state-machine merged in")

    def _parameters_changed(self, parameters):
        changed = {}
        for parameter in parameters:
            if parameter.name in self.values:
                changed[parameter.name] = parameter.value
        self.values.update(changed); self.core.update_config(changed)
        return SetParametersResult(successful=True)

    def _decode(self, message):
        if isinstance(message, CompressedImage):
            return self.bridge.compressed_imgmsg_to_cv2(message, desired_encoding="bgr8")
        return self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    def _publish_jpeg(self, publisher, frame, header=None) -> None:
        try:
            quality = max(10, min(100, int(self.values["debug.jpeg_quality"])))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return
            message = CompressedImage(); message.format = "jpeg"; message.data = encoded.tobytes()
            if header is not None:
                message.header = header
            publisher.publish(message)
        except (cv2.error, ValueError) as error:
            self.get_logger().warn(f"debug JPEG conversion failed: {error}", throttle_duration_sec=2.0)

    def _open_gate(self, reason: str) -> None:
        """출발 게이트를 연다(1회성). 자동(GREEN 확정) 경로와 수동(t키) 경로가
        공유한다 - 둘 다 게이트 해제 + 주행용 ROI 확장을 동일하게 해야 하므로."""
        if self.gate_opened:
            return
        self.gate_opened = True
        self.get_logger().info("출발 게이트 해제(1회성) - {}".format(reason))
        # 출발했으니 이제부터는 주행용 넓은 ROI로 전환(원근 때문에 좁은
        # ROI로는 접근 중 신호등을 놓치는 문제 방지) - 한 번만 전환.
        self.core.update_config({
            "search_roi.y": POST_GATE_ROI_Y,
            "search_roi.h": POST_GATE_ROI_H,
        })
        self.get_logger().info(
            "ROI를 주행용으로 확장: y=[{:.2f}~{:.2f}]".format(
                POST_GATE_ROI_Y, POST_GATE_ROI_Y + POST_GATE_ROI_H))

    def _on_force_gate(self, msg: Bool) -> None:
        if bool(msg.data):
            self._open_gate("수동(MUX t키) 강제 통과")

    def _check_gate_timeout(self) -> None:
        if self.gate_opened:
            self.gate_timeout_timer.cancel()
            return
        if time.monotonic() - self._node_start_time >= GATE_TIMEOUT_SEC:
            self._open_gate("{:.0f}초 타임아웃(자동 안전장치)".format(GATE_TIMEOUT_SEC))
            self.gate_timeout_timer.cancel()

    def _update_state_machine(self, straight_visible: bool, left_visible: bool):
        now = time.monotonic()

        self.straight_streak = self.straight_streak + 1 if straight_visible else 0
        straight_confirmed = self.straight_streak >= CONFIRM_FRAMES
        red_recent = self.frames_since_red < RED_BLOCK_FRAMES
        if not self.gate_opened and straight_confirmed:
            if red_recent:
                self.get_logger().warn(
                    "GREEN 확정됐지만 최근 {}프레임 안에 RED도 보여서 불안정 판정으로 보고 게이트 보류"
                    .format(RED_BLOCK_FRAMES))
            else:
                self._open_gate("GREEN(STRAIGHT) {}프레임 연속 확인".format(CONFIRM_FRAMES))

        # LEFT 등장 디바운스: CONFIRM_FRAMES 연속 보여야 "확정 감지"로 인정.
        self.left_streak = self.left_streak + 1 if left_visible else 0
        if self.left_streak >= CONFIRM_FRAMES:
            self.left_confirmed_visible = True
            self.left_absent_streak = 0
        elif self.left_confirmed_visible:
            # 확정된 상태에서 미검출 프레임이 이어지는 중 - 사라짐도 디바운스한다.
            if left_visible:
                self.left_absent_streak = 0
            else:
                self.left_absent_streak += 1
                if self.left_absent_streak >= LEFT_ABSENT_FRAMES:
                    self.left_until = now + LEFT_DURATION_SEC
                    self.get_logger().info(
                        "LEFT {}프레임 연속 미검출 -> 좌회전 발동 (재발동 가능)".format(LEFT_ABSENT_FRAMES))
                    self.left_confirmed_visible = False
                    self.left_absent_streak = 0

        if self.left_until is not None:
            if now < self.left_until:
                return True, LEFT_ANGLE, LEFT_SPEED
            # 좌회전 지속시간이 막 끝나는 순간 - 조향 정렬 단계로 전환
            self.straighten_until = now + STRAIGHTEN_DURATION_SEC
            self.left_until = None

        if self.straighten_until is not None:
            if now < self.straighten_until:
                return True, 0.0, LEFT_SPEED
            self.straighten_until = None

        if not self.gate_opened:
            return True, 0.0, 0.0

        return False, 0.0, 0.0

    def _image_callback(self, message) -> None:
        self.last_image_wall = time.monotonic(); self.stale_published = False
        try:
            frame = self._decode(message)
        except (CvBridgeError, cv2.error, ValueError, TypeError) as error:
            self.get_logger().warn(f"image conversion failed: {error}", throttle_duration_sec=2.0)
            return
        if frame is None or frame.size == 0:
            self.get_logger().warn("empty camera frame ignored", throttle_duration_sec=2.0)
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        try:
            result = self.core.detect(frame, stamp=stamp, make_overlay=bool(
                self.values["debug.publish_images"]))
        except (cv2.error, ValueError, IndexError, TypeError) as error:
            self.get_logger().error(f"detector recovered from frame error: {error}",
                                    throttle_duration_sec=1.0)
            return
        self.state_pub.publish(String(data=result.state))
        score_message = Float32MultiArray()
        score_message.data = [float(result.scores.get(name, 0.0)) for name in STATES] + [
            float(result.detection_confidence), float(result.white_light_score)]
        self.score_pub.publish(score_message)

        # mux 연동 상태머신 (구 traffic_light_bridge.py)
        state = result.state
        straight_visible = state == "GREEN" and not self.gate_opened
        left_visible = state == "LEFT"
        self.frames_since_red = 0 if state == "RED" else self.frames_since_red + 1
        active, angle, speed = self._update_state_machine(straight_visible, left_visible)
        self.last_active, self.last_angle, self.last_speed = active, angle, speed
        self.active_pub.publish(Bool(data=active))
        if active:
            self.cmd_pub.publish(Float32MultiArray(data=[angle, speed]))

        if bool(self.values["debug.publish_images"]):
            if result.overlay is not None:
                # 리사이즈+텍스트 그리기(무거움)는 여기서 매 프레임 하지 않고, 실제
                # 화면 표시 주기(10Hz, _render_timer)에서만 한다 - 카메라 fps 그대로
                # 매번 하면 표시 안 하는 프레임에도 비용을 치르게 되어 CPU 낭비.
                self._latest_overlay = result.overlay
                self._latest_state = state
            mask = result.masks.get("morphology")
            if mask is not None:
                self._publish_jpeg(self.mask_pub, mask, message.header)

    def _render_timer(self) -> None:
        # t키: 출발 게이트 수동 강제 통과(1회성). GREEN을 못 보거나 테스트 중
        # 매번 기다리기 싫을 때 씀 - imshow 전에 검사해야 창에 프레임이 아직
        # 하나도 안 떴어도(카메라 시작 직후 등) 키를 놓치지 않는다.
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('t'), ord('T')):
            self._open_gate("수동(t키) 강제 통과")
        if self._latest_overlay is None:
            return
        dbg = cv2.resize(self._latest_overlay, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                         interpolation=cv2.INTER_LINEAR)
        dh = dbg.shape[0]
        cv2.putText(dbg, "state:{}".format(self._latest_state),
                   (10, dh - 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        cv2.putText(dbg, "gate:{}  active:{} ({:+.0f},{:.0f})  frames_since_red:{}".format(
                   int(self.gate_opened), int(self.last_active),
                   self.last_angle, self.last_speed,
                   min(self.frames_since_red, 999)),
                   (10, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.putText(dbg, "t:gate force-open",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        cv2.imshow(WIN, dbg)

    def _check_input_timeout(self) -> None:
        timeout = max(0.1, float(self.values["input_timeout_sec"]))
        if not self.stale_published and time.monotonic() - self.last_image_wall > timeout:
            self.state_pub.publish(String(data="UNKNOWN")); self.stale_published = True
            self.core.reset()
            self.get_logger().warn("camera topic is stale; published UNKNOWN",
                                   throttle_duration_sec=5.0)


def main(args=None) -> None:
    rclpy.init(args=args); node = TrafficLightDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
