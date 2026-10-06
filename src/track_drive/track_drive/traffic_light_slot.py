#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# traffic_light_slot.py - YOLO26n(ONNX)은 신호등 "위치(박스)"만 찾는 용도로 쓰고,
#   실제 상태(직진/좌회전) 판정은 클래스 예측을 아예 쓰지 않고 박스를 4등분한
#   슬롯의 밝기(점등 여부)로 규칙 기반 판정한다. (traffic_light_yolo.py의 클래스
#   분류가 빨간불을 좌회전으로 오인하는 문제 때문에, traffic_light_detector 패키지의
#   slot_boundaries=[0,0.25,0.5,0.75,1.0] 4슬롯 방식을 참고해서 새로 만든 버전)
#
#   규칙: 박스를 좌->우 4등분 슬롯으로 나눠서 각 슬롯이 켜져있는지(밝기 기준) 본다.
#     - 우측 2개 슬롯(슬롯3,4)이 둘 다 켜짐 -> STRAIGHT_LEFT
#     - 최우측 1개 슬롯(슬롯4)만 켜짐 -> STRAIGHT
#     - 그 외 조합은 NONE (빨강/노랑 등 좌측 슬롯 점등은 이 노드가 취급하는 범위 밖)
#
#   기존 traffic_light_yolo.py는 그대로 두고 별도 노드로 만듦 - 토픽은 동일하게
#   /cmd/traffic_light, /traffic_light/active 발행이라 mux 연동 그대로 재사용 가능.

import time
import rclpy, cv2
import numpy as np
import onnxruntime as ort
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Float32MultiArray
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

MODEL = '/home/user/ruby_ws/traffic_light_yolo/yolo26n_traffic_light_v2.onnx'
IMG_W, IMG_H = 640, 480
IN_SIZE = 320
INFER_EVERY = 3
BOX_CONF_TH = 0.35  # 박스(위치) 검출 신뢰도 - 클래스 예측 자체는 무시하고 위치만 씀

# ROI: traffic_light_yolo.py와 동일 (실측 박스 분포 기준 캘리브레이션됨)
ROI_X0, ROI_X1 = 0.15, 0.85
ROI_Y0, ROI_Y1 = 0.00, 0.55

# 박스를 좌->우 4등분하는 슬롯 경계 (정규화, traffic_light_detector 참고)
SLOT_BOUNDARIES = [0.0, 0.25, 0.50, 0.75, 1.0]
# 슬롯 점등 판정: HSV에서 이 정도 이상 밝고(V) 채도 있는(S) 픽셀 비율이 넘으면 "켜짐"
# traffic_light_detector 패키지의 traffic_light.yaml에서 실제 캘리브레이션된 값
# (red1/red2/yellow/green + 흰색 코어) 그대로 가져옴. 색상 구분은 안 하고 이 중
# 아무 범위에나 걸리면 "점등"으로 봄(합집합 마스크).
HSV_LIT_RANGES = [
    ((0,   55, 75),  (12,  255, 255)),   # red1
    ((168, 55, 75),  (179, 255, 255)),   # red2
    ((12,  45, 90),  (38,  255, 255)),   # yellow
    ((38,  35, 70),  (102, 255, 255)),   # green
]
WHITE_V_MIN, WHITE_S_MAX = 238, 48        # 점등 램프의 흰색 포화 코어
SLOT_LIT_RATIO_TH = 0.045                 # halo.min_color_ratio와 동일

CLASS_STRAIGHT = 0
CLASS_STRAIGHT_LEFT = 1

LEFT_ANGLE = -30.0
LEFT_SPEED = 20.0
LEFT_DURATION_SEC = 1.0
CONFIRM_FRAMES = 3  # 이 프레임 연속 검출돼야 진짜로 인정 (오탐 방지 디바운스)
LEFT_ABSENT_FRAMES = 3  # 확정된 LEFT가 이만큼 연속 미검출돼야 "진짜 사라짐"으로 인정

WIN = 'TL-Slot'
DISPLAY_SCALE = 1.6  # 디버그 창을 이 배율로 키워서 표시 (화면 작아서 글씨 안 보이는 문제)


def _letterbox_square(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    return canvas, scale, pad_x, pad_y


def slot_lit_states(crop):
    """크롭된 박스 영역을 4등분해서 각 슬롯의 점등 여부(bool 4개)를 반환."""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return [False, False, False, False]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (179, WHITE_S_MAX, 255))  # 흰색 코어
    for lo, hi in HSV_LIT_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    lit = []
    for i in range(4):
        x0 = int(round(SLOT_BOUNDARIES[i] * w))
        x1 = int(round(SLOT_BOUNDARIES[i + 1] * w))
        if x1 <= x0:
            lit.append(False)
            continue
        slot_mask = mask[:, x0:x1]
        ratio = float((slot_mask > 0).mean())
        lit.append(ratio >= SLOT_LIT_RATIO_TH)
    return lit


def classify_by_slots(lit):
    """lit = [슬롯1,2,3,4] 점등여부. 우측 2개(3,4) 켜짐->STRAIGHT_LEFT,
    최우측 1개(4)만 켜짐->STRAIGHT, 그 외 NONE."""
    if lit[2] and lit[3]:
        return 'STRAIGHT_LEFT'
    if lit[3] and not lit[2]:
        return 'STRAIGHT'
    return 'NONE'


class TrafficLightSlot(Node):

    def __init__(self):
        super().__init__('traffic_light_slot')
        self.bridge = CvBridge()

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(MODEL, sess_options=so,
                                          providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name

        self.frame_n = 0
        self.last_box = None       # (x1,y1,x2,y2) ROI 좌표계, 없으면 None
        self.last_lit = [False] * 4
        self.last_state = 'NONE'
        self.last_infer_ms = 0.0
        self.n_infer = 0
        self.vis = None

        self.gate_opened = False
        self.straight_streak = 0
        self.left_streak = 0
        self.left_confirmed_visible = False
        self.left_absent_streak = 0
        self.left_until = None

        self.pub_state = self.create_publisher(String, '/state/traffic_state_slot', 10)
        self.pub_cmd = self.create_publisher(Float32MultiArray, '/cmd/traffic_light', 10)
        self.pub_active = self.create_publisher(Bool, '/traffic_light/active', 10)
        self.create_subscription(Image, '/usb_cam/image_raw/front',
                                 self.cb, qos_profile_sensor_data)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, int(IMG_W * DISPLAY_SCALE), int(IMG_H * DISPLAY_SCALE))
        self.create_timer(0.1, self._render_timer)
        self.get_logger().info(
            'traffic_light_slot ready | infer_every={} imgsz={} model={}'.format(
                INFER_EVERY, IN_SIZE, MODEL))

    def _roi_crop(self, img):
        h, w = img.shape[:2]
        x0, x1 = int(w * ROI_X0), int(w * ROI_X1)
        y0, y1 = int(h * ROI_Y0), int(h * ROI_Y1)
        return img[y0:y1, x0:x1], (x0, y0)

    def _detect_box(self, roi):
        """가장 신뢰도 높은 박스 하나만 위치용으로 반환 (클래스는 무시)."""
        sq, scale, pad_x, pad_y = _letterbox_square(roi, IN_SIZE)
        x = sq.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        x = x[:, ::-1, :, :].copy()

        t0 = time.monotonic()
        out = self.sess.run(None, {self.input_name: x})[0]
        self.last_infer_ms = (time.monotonic() - t0) * 1000.0
        self.n_infer += 1

        dets = out[0]  # (300, 6) = [x1, y1, x2, y2, score, cls]
        cls_id = dets[:, 5]
        conf = dets[:, 4]
        mask = (conf >= BOX_CONF_TH) & ((cls_id == CLASS_STRAIGHT) | (cls_id == CLASS_STRAIGHT_LEFT))
        if not np.any(mask):
            return None
        cand = dets[mask]
        best = cand[np.argmax(cand[:, 4])]
        bx1, by1, bx2, by2 = best[:4]
        bx1 = (bx1 - pad_x) / scale
        by1 = (by1 - pad_y) / scale
        bx2 = (bx2 - pad_x) / scale
        by2 = (by2 - pad_y) / scale
        return (bx1, by1, bx2, by2)

    def _update_state_machine(self, straight_visible, left_visible):
        now = time.monotonic()

        # 디바운스: CONFIRM_FRAMES 연속 검출돼야 "진짜 감지"로 인정.
        # 순간적인 오탐지 한 프레임만으로 게이트가 풀리거나 좌회전이 발동하는 걸 막는다.
        self.straight_streak = self.straight_streak + 1 if straight_visible else 0
        straight_confirmed = self.straight_streak >= CONFIRM_FRAMES
        if not self.gate_opened and straight_confirmed:
            self.gate_opened = True
            self.get_logger().info(
                'STRAIGHT {}프레임 연속 확인 -> 출발 게이트 해제(1회성)'.format(CONFIRM_FRAMES))

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
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        img = cv2.resize(img, (IMG_W, IMG_H))
        roi, (ox, oy) = self._roi_crop(img)

        self.frame_n += 1
        if self.frame_n % INFER_EVERY == 0:
            box = self._detect_box(roi)
            self.last_box = box
            if box is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(roi.shape[1], x2)
                y2 = min(roi.shape[0], y2)
                crop = roi[y1:y2, x1:x2]
                self.last_lit = slot_lit_states(crop)
                self.last_state = classify_by_slots(self.last_lit)
            else:
                self.last_lit = [False] * 4
                self.last_state = 'NONE'
            self.pub_state.publish(String(data=self.last_state))

        straight_visible = self.last_state == 'STRAIGHT'
        left_visible = self.last_state == 'STRAIGHT_LEFT'
        active, angle, speed = self._update_state_machine(straight_visible, left_visible)
        self.pub_active.publish(Bool(data=active))
        if active:
            self.pub_cmd.publish(Float32MultiArray(data=[angle, speed]))

        dbg = img.copy()
        rx1 = int(img.shape[1] * ROI_X1)
        ry1 = int(img.shape[0] * ROI_Y1)
        cv2.rectangle(dbg, (ox, oy), (rx1, ry1), (0, 255, 255), 1)
        if self.last_box is not None:
            x1, y1, x2, y2 = self.last_box
            p1 = (int(ox + x1), int(oy + y1))
            p2 = (int(ox + x2), int(oy + y2))
            cv2.rectangle(dbg, p1, p2, (0, 0, 255), 2)
            # 슬롯 경계선 + 점등 표시
            bw = x2 - x1
            for i in range(4):
                sx0 = int(ox + x1 + SLOT_BOUNDARIES[i] * bw)
                sx1 = int(ox + x1 + SLOT_BOUNDARIES[i + 1] * bw)
                col = (0, 255, 0) if self.last_lit[i] else (90, 90, 90)
                cv2.rectangle(dbg, (sx0, p1[1]), (sx1, p2[1]), col, 2)

        # 화면이 작아서 글씨가 안 보이는 문제 - 최종 표시 직전에 캔버스 자체를
        # DISPLAY_SCALE배 키운 다음 그 위에 크고 두꺼운 글씨를 얹는다.
        dbg = cv2.resize(dbg, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                          interpolation=cv2.INTER_LINEAR)
        dh = dbg.shape[0]
        cv2.putText(dbg, '{}  lit:{}'.format(self.last_state, [int(v) for v in self.last_lit]),
                    (10, dh - 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        cv2.putText(dbg, 'infer:{:.1f}ms  gate:{}  active:{} ({:+.0f},{:.0f})'.format(
                    self.last_infer_ms, int(self.gate_opened), int(active), angle, speed),
                    (10, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        self.vis = dbg

    def _render_timer(self):
        if self.vis is not None:
            cv2.imshow(WIN, self.vis)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightSlot()
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
