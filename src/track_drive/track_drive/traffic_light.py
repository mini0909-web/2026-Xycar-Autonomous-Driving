#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# traffic_light.py - 중앙 신호등 상태 판정 노드 (HSV + 모양 기반, 경찰차는 안 봄)
#   /usb_cam/image_raw/front 를 구독해 화면 상단의 점등 램프를 직접 검출한다.
#   핵심: 점등 램프는 고채도(S>=150) -> 나무/잔디(저채도) 자동 제거.
#         초록은 채움 비율(fill)로 모양 판정: 원반=직진 / 화살표=좌회전.
#   발행 상태: NONE / RED / YELLOW / STRAIGHT / LEFT / RED_LEFT  (-> /state/traffic_state)
#   진행 의도(직진/좌회전)는 e2e_drive 가 자기 angle 로 판단하므로 경찰차 검출은 불필요.
# ============================================================================

import rclpy, cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

# 검출 파라미터 (071736 데이터셋으로 캘리브레이션)
#   S_MIN,V_MIN: 점등 램프(발광)의 고채도+밝음 기준(잔디/나무 제거)
#   AREA/AR: 램프 bbox 면적·종횡비(대략 원형) / FILL_DISC: 초록 원반(직진) vs 화살표(좌회전)
#   DISC_FILL_MIN: 빨/노는 원반만 인정 / TOP: 신호등은 상단부 / DEBOUNCE: 깜빡임 방지
S_MIN, V_MIN = 150, 110
AREA_MIN, AREA_MAX = 110, 6000
AR_LO, AR_HI = 0.4, 2.6
FILL_DISC = 0.80
DISC_FILL_MIN = 0.50
TOP = 0.45
DEBOUNCE = 3

# 색상별 HSV 범위(빨강은 hue 양끝 두 구간) / 디버그 표시 색
RANGES = {
    'red':    [((0,   S_MIN, V_MIN), (10,  255, 255)),
               ((165, S_MIN, V_MIN), (180, 255, 255))],
    'yellow': [((11,  S_MIN, V_MIN), (35,  255, 255))],
    'green':  [((40,  S_MIN, V_MIN), (90,  255, 255))],
}
DRAW = {'red': (0, 0, 255), 'yellow': (0, 255, 255),
        'straight': (0, 255, 0), 'left': (255, 128, 0)}


# 신호등 판정 노드: 카메라 상단 ROI에서 점등 램프를 찾아 신호 상태를 발행한다.
class TrafficLightNode(Node):

    def __init__(self):
        # 노드 초기화: 상태/디바운스 변수 준비, 신호 발행자·카메라 구독 설정
        super().__init__('traffic_light')
        self.bridge = CvBridge()
        self.state = 'NONE'
        self.cand = 'NONE'
        self.cand_n = 0
        self.pub = self.create_publisher(String, '/state/traffic_state', 10)
        self.create_subscription(Image, '/usb_cam/image_raw/front',
                                 self.cam_cb, qos_profile_sensor_data)
        self.get_logger().info('traffic_light ready -> /state/traffic_state')

    def _blobs(self, m):
        # 색 마스크에서 램프 후보 추출: 잡음 제거 후 면적·종횡비 조건을 통과한
        #   덩어리만 (x, y, w, h, 채움비율)로 반환한다.
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < AREA_MIN or area > AREA_MAX:
                continue
            ar = w / float(h)
            if ar < AR_LO or ar > AR_HI:
                continue
            fill = (m[y:y+h, x:x+w] > 0).mean()
            out.append((x, y, w, h, fill))
        return out

    def classify(self, img):
        # 신호 판정: 상단 ROI에서 색별로 램프를 찾아 어떤 등이 켜졌는지 종합한다.
        #   초록은 채움 비율로 직진/좌회전 구분, 빨/노는 원반만 인정.
        h, w = img.shape[:2]
        roi = img[0:int(h * TOP), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        on = {'red': False, 'yellow': False, 'straight': False, 'left': False}
        dbg = roi.copy()

        # 색별 마스크 -> 램프 후보 -> 켜짐 플래그 갱신 + 디버그 박스
        for color in ('red', 'yellow', 'green'):
            m = None
            for lo, hi in RANGES[color]:
                mm = cv2.inRange(hsv, np.array(lo), np.array(hi))
                m = mm if m is None else (m | mm)
            for (x, y, bw, bh, fill) in self._blobs(m):
                if color == 'green':
                    tag = 'straight' if fill >= FILL_DISC else 'left'
                    on[tag] = True
                else:
                    if fill < DISC_FILL_MIN:
                        continue
                    tag = color
                    on[color] = True
                cv2.rectangle(dbg, (x, y), (x+bw, y+bh), DRAW[tag], 1)
                cv2.putText(dbg, '{}{:.2f}'.format(tag[:1], fill), (x, max(9, y-2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, DRAW[tag], 1)

        # 켜진 등 조합 -> 최종 상태 문자열 결정(우선순위 적용)
        if on['straight']:               st = 'STRAIGHT'
        elif on['red'] and on['left']:   st = 'RED_LEFT'
        elif on['left']:                 st = 'LEFT'
        elif on['red']:                  st = 'RED'
        elif on['yellow']:               st = 'YELLOW'
        else:                            st = 'NONE'
        return st, dbg

    def cam_cb(self, msg):
        # 카메라 콜백: 한 프레임 판정 -> 디바운스로 확정 -> 상태 발행 + 디버그 창 표시
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        raw, dbg = self.classify(img)

        # 디바운스: 같은 후보가 DEBOUNCE 프레임 연속일 때만 상태 확정(깜빡임 방지)
        if raw == self.cand:
            self.cand_n += 1
        else:
            self.cand, self.cand_n = raw, 1
        if self.cand_n >= DEBOUNCE and self.state != self.cand:
            self.state = self.cand
            self.get_logger().info('signal -> {}'.format(self.state))

        # 현재 상태 발행 + 디버그 영상에 상태 글자 표시
        self.pub.publish(String(data=self.state))
        cv2.putText(dbg, self.state, (3, dbg.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('TL-DBG', dbg)
        cv2.waitKey(1)


def main(args=None):
    # 노드 실행 진입점: 초기화 -> spin(콜백 루프) -> 종료 정리
    rclpy.init(args=args)
    node = TrafficLightNode()
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
