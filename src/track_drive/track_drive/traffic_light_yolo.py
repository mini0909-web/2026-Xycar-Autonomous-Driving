#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# traffic_light_yolo.py - YOLO26n(ONNX) 기반 경량 신호등 검출 + 상태머신 -> mux 연동
#   MODEL v2: kmu 신호등 데이터셋(Job3+Job4, 1986장) 재학습본. class 0=straight,
#   class 1=straight_left (mAP50 ~0.995, precision ~0.99, recall ~0.99~1.0).
#   v1 대비 변경점: fliplr=0(좌우반전 증강 끔 - straight_left가 방향성 있는 클래스라
#   반전하면 라벨과 실제 화살표 방향이 모순되던 문제 수정), hsv 증강은 유지.
#   그래도 실제 오탐지율은 처음 트랙 실주행에서 확인 필요 - start_motor:=false로
#   먼저 디버그 화면 보면서 게이트/좌회전 트리거가 원하는 지점에서만 뜨는지 확인할 것.
#
#   상태머신 (mux_node.py가 최우선순위로 소비 - /cmd/traffic_light, /traffic_light/active):
#     1) 시작 게이트: class STRAIGHT(0)가 처음 감지될 때까지 speed=0 강제. 한 번 감지되면
#        그 뒤로 이 클래스가 보이든 안 보이든 무관하게 영구적으로 게이트 해제(1회성).
#     2) 좌회전: class STRAIGHT_LEFT(1)가 보이다가 사라지는 순간(falling edge)마다
#        angle=-75, speed=20을 LEFT_DURATION_SEC(1초)간 강제. 재발동 가능(1회성 아님) -
#        트랙에 이 구간이 여러 곳일 수 있어서 매번 다시 트리거되게 함.
#     오탐지 방지(디바운스)는 실제 학습모델+실제 트랙 데이터로 나중에 튜닝 예정.
#
#   경량화 포인트:
#     - 화면 전체가 아니라 고정 ROI(신호등이 나오는 구간)만 잘라서 추론
#     - INFER_EVERY 프레임마다 1번만 추론 (나머지 프레임은 직전 결과 유지)
#     - 입력 해상도 320x320 (yolo26n 기본 export 크기)
#     - onnxruntime CPUExecutionProvider, intra-op 스레드 1로 고정
#       (작은 모델을 매 프레임 돌릴 땐 멀티스레드 오버헤드가 더 큼 - e2e_pure와 동일 이유)

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
IN_SIZE = 320                    # 모델 입력 정사각형 크기 (export 시 imgsz=320)
INFER_EVERY = 2                  # 이 프레임마다 1번만 추론
CONF_TH = 0.7

# 커스텀 학습 클래스 (실제 신호등 모델 기준)
CLASS_STRAIGHT = 0
CLASS_STRAIGHT_LEFT = 1

LEFT_ANGLE = -30.0
LEFT_SPEED = 20.0
LEFT_DURATION_SEC = 1.0
CONFIRM_FRAMES = 3  # 이 프레임 연속 검출돼야 진짜로 인정 (오탐 방지 디바운스)
LEFT_ABSENT_FRAMES = 3  # 확정된 LEFT가 이만큼 연속 미검출돼야 "진짜 사라짐"으로 인정

# ROI: 신호등이 화면에서 나오는 대략적인 구간 (실제 신호등 이미지로 나중에 캘리브레이션 필요)
ROI_X0, ROI_X1 = 0.15, 0.85      # 좌우 비율
ROI_Y0, ROI_Y1 = 0.00, 0.55      # 상하 비율 (상단부)

WIN = 'TL-YOLO'
DISPLAY_SCALE = 1.6  # 디버그 창을 이 배율로 키워서 표시 (화면 작아서 글씨 안 보이는 문제)


def _letterbox_square(img, size):
    """정사각형 패딩 후 리사이즈. 반환: (정사각형 이미지, scale, pad_x, pad_y)"""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    return canvas, scale, pad_x, pad_y


class TrafficLightYolo(Node):

    def __init__(self):
        super().__init__('traffic_light_yolo')
        self.bridge = CvBridge()

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(MODEL, sess_options=so,
                                          providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name

        self.frame_n = 0
        self.last_state = 'NONE'
        self.last_boxes = []
        self.last_straight_visible = False
        self.last_left_visible = False
        self.last_infer_ms = 0.0
        self.n_infer = 0
        self.vis = None

        # 상태머신 상태
        self.gate_opened = False          # STRAIGHT(0) 최초 감지 시 영구 True (1회성)
        self.straight_streak = 0
        self.left_streak = 0
        self.left_confirmed_visible = False
        self.left_absent_streak = 0
        self.left_until = None

        self.pub_state = self.create_publisher(String, '/state/traffic_state_yolo', 10)
        self.pub_cmd = self.create_publisher(Float32MultiArray, '/cmd/traffic_light', 10)
        self.pub_active = self.create_publisher(Bool, '/traffic_light/active', 10)
        self.create_subscription(Image, '/usb_cam/image_raw/front',
                                 self.cb, qos_profile_sensor_data)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, int(IMG_W * DISPLAY_SCALE), int(IMG_H * DISPLAY_SCALE))
        self.create_timer(0.1, self._render_timer)
        self.get_logger().info(
            'traffic_light_yolo ready | infer_every={} imgsz={} model={}'.format(
                INFER_EVERY, IN_SIZE, MODEL))

    def _roi_crop(self, img):
        h, w = img.shape[:2]
        x0, x1 = int(w * ROI_X0), int(w * ROI_X1)
        y0, y1 = int(h * ROI_Y0), int(h * ROI_Y1)
        return img[y0:y1, x0:x1], (x0, y0)

    def _infer(self, roi):
        sq, scale, pad_x, pad_y = _letterbox_square(roi, IN_SIZE)
        x = sq.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]  # NCHW, BGR (COCO 사전학습 기준 RGB 필요시 별도 변환)
        x = x[:, ::-1, :, :].copy()  # BGR->RGB

        t0 = time.monotonic()
        out = self.sess.run(None, {self.input_name: x})[0]  # (1, 300, 6): NMS 내장(end-to-end)
        self.last_infer_ms = (time.monotonic() - t0) * 1000.0
        self.n_infer += 1

        dets = out[0]  # (300, 6) = [x1, y1, x2, y2, score, cls]
        cls_id = dets[:, 5]
        conf = dets[:, 4]

        straight_visible = bool(np.any((conf >= CONF_TH) & (cls_id == CLASS_STRAIGHT)))
        left_visible = bool(np.any((conf >= CONF_TH) & (cls_id == CLASS_STRAIGHT_LEFT)))

        mask = (conf >= CONF_TH) & ((cls_id == CLASS_STRAIGHT) | (cls_id == CLASS_STRAIGHT_LEFT))
        boxes = []
        for bx1, by1, bx2, by2, sc, cid in dets[mask]:
            # letterbox 역변환: 패딩 제거 후 스케일 복원 (ROI 좌표계 기준)
            bx1 = (bx1 - pad_x) / scale
            by1 = (by1 - pad_y) / scale
            bx2 = (bx2 - pad_x) / scale
            by2 = (by2 - pad_y) / scale
            boxes.append((bx1, by1, bx2, by2, float(sc), int(cid)))
        return straight_visible, left_visible, boxes

    def _update_state_machine(self, straight_visible, left_visible):
        """상태머신 갱신. 반환: (active, angle, speed)"""
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
            straight_visible, left_visible, self.last_boxes = self._infer(roi)
            self.last_straight_visible = straight_visible
            self.last_left_visible = left_visible
            self.last_state = ('STRAIGHT' if straight_visible else
                                'STRAIGHT_LEFT' if left_visible else 'NONE')
            self.pub_state.publish(String(data=self.last_state))

        # 상태머신은 추론 여부와 무관하게 매 프레임 갱신 - 원샷 좌회전의 1초 타이밍과
        # mux로의 활성 신호 최신성(staleness 체크)을 프레임 단위로 유지하기 위함.
        active, angle, speed = self._update_state_machine(
            self.last_straight_visible, self.last_left_visible)
        self.pub_active.publish(Bool(data=active))
        if active:
            self.pub_cmd.publish(Float32MultiArray(data=[angle, speed]))

        dbg = img.copy()
        rx0, ry0 = ox, oy
        rx1 = int(img.shape[1] * ROI_X1)
        ry1 = int(img.shape[0] * ROI_Y1)
        cv2.rectangle(dbg, (rx0, ry0), (rx1, ry1), (0, 255, 255), 1)
        for (bx1, by1, bx2, by2, sc, cid) in self.last_boxes:
            p1 = (int(rx0 + bx1), int(ry0 + by1))
            p2 = (int(rx0 + bx2), int(ry0 + by2))
            col = (0, 255, 0) if cid == CLASS_STRAIGHT else (255, 128, 0)
            cv2.rectangle(dbg, p1, p2, col, 2)
            cv2.putText(dbg, '{}:{:.2f}'.format(cid, sc), (p1[0], max(9, p1[1]-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # 화면이 작아서 글씨가 안 보이는 문제 - 최종 표시 직전에 캔버스 자체를
        # DISPLAY_SCALE배 키운 다음 그 위에 크고 두꺼운 글씨를 얹는다.
        dbg = cv2.resize(dbg, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                          interpolation=cv2.INTER_LINEAR)
        dh = dbg.shape[0]
        cv2.putText(dbg, '{}  infer:{:.1f}ms  n_infer:{}'.format(
                    self.last_state, self.last_infer_ms, self.n_infer),
                    (10, dh - 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        cv2.putText(dbg, 'gate_opened:{}  active:{} ({:+.0f},{:.0f})'.format(
                    int(self.gate_opened), int(active), angle, speed),
                    (10, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        self.vis = dbg

    def _render_timer(self):
        if self.vis is not None:
            cv2.imshow(WIN, self.vis)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightYolo()
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
