#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# e2e_drive.py - PilotNet 기반 End-to-End 자율주행 노드
#   동작 한 줄 요약: 전방 카메라 한 장 -> PilotNet 추론 -> [조향, 속도] -> 모터
#   학습 산출물: /home/user/xycar_ws/e2e/model.pth(가중치) + norm.json(라벨 정규화)
#   * 전처리(상단크롭 -> 200x66 -> YUV)는 학습(train.py)과 반드시 동일해야 함 *
#
#   [신호등 게이트 개요]
#   /state/traffic_state(String) 구독: NONE/RED/YELLOW/STRAIGHT/LEFT/RED_LEFT
#   - 진행 의도는 모델이 내는 angle 로 판단(크게 좌(-)면 좌회전, 그 외 직진)
#   - 초록 켜짐 -> GO / 단독 RED·YELLOW -> STOP / NONE -> 평상 주행
#   - 적용 시점: 첫 신호등은 신호로만, 이후엔 정지선 보일 때만, GO 후 grace 동안 게이트 끔
#
#   [추가 주행 규칙(이 프로젝트에서 보강)]
#   - 신호등 통과 후 일정 구간 속도 고정(FIX_*)
#   - 바닥 양쪽이 노란선이면 속도 고정(YL_*), 끝나면 다시 모델 추론 속도로 복귀
# ============================================================================

import os, json, time
import rclpy, cv2
import numpy as np
import torch
import torch.nn as nn
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

# 학습 산출물 경로: 가중치(model.pth)와 라벨 정규화 파라미터(norm.json)
MODEL = '/home/user/ruby_ws/e2e/model.pth'
NORM  = '/home/user/ruby_ws/e2e/norm.json'

# 입력 영상/전처리 규격 (train.py 와 동일해야 함)
#   IMG: 원본 처리 해상도 / IN: PilotNet 입력 / CROP_TOP: 상단 40%(하늘) 제거
#   MAX_ANGLE: 조향 한계 / MIN_SPEED: GO 일 때 하한 속도(정지 분기엔 미적용)
#   INTENT_LEFT_ANGLE: angle 이 이 값보다 작으면 '좌회전 의도'로 판정
IMG_W, IMG_H = 640, 480
IN_W, IN_H   = 200, 66
CROP_TOP     = 0.40
MAX_ANGLE    = 100.0
MIN_SPEED    = 10.0
MAX_SPEED    = 15.0
INTENT_LEFT_ANGLE = -25.0

# 바닥 양쪽 노란선 감지 -> 속도 고정 파라미터
#   차 좌/우 노면 ROI 가 모두 노란색이면 YL_SPEED 로 고정, 끝나면 추론 속도 복귀.
#   YL_LO/HI: 노란색 HSV 범위 / YL_Y*: 노면 세로 ROI / YL_LX*,RX*: 좌·우 가로 ROI
#   YL_COV: 각 ROI 에서 노란 픽셀 비율 임계(이상이면 노란선 있음)
YL_SPEED   = 16.5
YL_LO = np.array([15,  80,  80])
YL_HI = np.array([40, 255, 255])
YL_Y0, YL_Y1   = 0.65, 0.98
YL_LX0, YL_LX1 = 0.00, 0.35
YL_RX0, YL_RX1 = 0.65, 1.00
YL_COV = 0.03

# 정지선(stop line) 게이트 파라미터
#   신호등은 '정지선이 보일 때만' 본다. 정지선 = 노면을 가로지르는 굵은 흰 가로 띠.
#   SL_Y*/X*: 노면 중앙 ROI(지평선/좌우 측선 제외) / SL_VW,SW: 흰색(고명도+저채도) 기준
#   SL_COV: '흰 행' 판정 비율 / SL_ON: 정지선 트리거 연속 행수
#   SL_LOW,OFF_N: 저검출(run<LOW)이 N프레임 연속이면 통과로 보고 게이트 해제
SL_Y0, SL_Y1 = 0.55, 0.95
SL_X0, SL_X1 = 0.18, 0.82
SL_VW, SL_SW = 180, 70
SL_COV       = 0.55
SL_ON        = 20
SL_LOW       = 3
SL_OFF_N     = 5

# 첫 신호등 / GO 후 grace / 통과 후 고정속도 파라미터
#   GRACE_SEC: GO 결정 후 이 시간 동안 신호 게이트를 꺼 멀어지는 신호등 오검출로 재정지 방지.
#   FIX_*: 신호등 통과 FIX_DELAY초 뒤부터 FIX_DUR초간 속도를 FIX_SPEED 로 고정.
GRACE_SEC    = 10.0
FIX_SPEED    = 17.0
FIX_DELAY    = 5.0
FIX_DUR      = 5.0
WIN = 'E2E Drive'


# PilotNet: NVIDIA End-to-End 자율주행 CNN.
#   합성곱 5층으로 화면 특징을 뽑고, 완전연결 4층으로 판단해 [조향, 속도] 2개를 낸다.
#   (원조 PilotNet 은 조향 1개만 출력 -> 출력층을 2개로 확장해 속도까지 회귀)
class PilotNet(nn.Module):
    def __init__(self):
        # 신경망 계층 정의: Conv 5층(특징 추출) -> Flatten/Dropout -> FC 4층(마지막 2출력)
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 1 * 18, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        # 순전파: 이미지 -> 합성곱 특징 -> 완전연결 -> [조향, 속도]
        return self.fc(self.conv(x))


# E2E 주행 노드.
#   전방 카메라 한 장을 받아 PilotNet 으로 조향·속도를 추론하고,
#   신호등/정지선/노란선 규칙을 반영한 최종 모터 명령을 발행한다.
class E2EDrive(Node):

    def __init__(self):
        # 노드 초기화: 모델·정규화 로드, 상태변수 준비, 토픽 구독/발행, 표시·진단 타이머 설정
        super().__init__('e2e_drive')
        self.bridge = CvBridge()

        # 모델 로드(GPU 있으면 cuda) 후 추론 모드로 전환, 역정규화 파라미터 로드
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(MODEL):
            self.get_logger().error('no model: {} (train.py 먼저)'.format(MODEL))
        self.net = PilotNet().to(self.dev)
        self.net.load_state_dict(torch.load(MODEL, map_location=self.dev))
        self.net.eval()
        self.norm = json.load(open(NORM))

        # 주행 상태 변수(추론 결과 / 신호·정지선 게이트 상태 / 카운터)
        self.angle = 0.0
        self.speed = 0.0
        self.vis = None
        self.n_cb = 0
        self.n_pub = 0
        self.signal = 'NONE'
        self.go = True
        self.intent = 'STRAIGHT'
        self.sl_active = False
        self.sl_low = 0
        self.sl_nband = 0
        self.first_done = False
        self.grace_until = 0.0
        self.light_pass_t = None
        self.yellow_lock = False

        self.pub_motor = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)

        # 전방 카메라(주행 추론) + 신호등 상태 구독, 표시 창과 주기 타이머 설정
        self.create_subscription(Image, '/usb_cam/image_raw/front',
                                 self.cb, qos_profile_sensor_data)
        self.create_subscription(String, '/state/traffic_state', self.sig_cb, 10)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, IMG_W, IMG_H)
        self.timer = self.create_timer(0.03, self.show)
        self.diag = self.create_timer(1.0, self.status)
        self.get_logger().info('e2e_drive ready dev={}'.format(self.dev))

    def sig_cb(self, msg):
        # 신호등 상태 콜백: traffic_light 노드가 보낸 신호 색 문자열을 저장
        self.signal = msg.data or 'NONE'

    def stopline_run(self, img):
        # 정지선 검출: 노면 중앙 ROI에서 연속된 흰 가로 띠의 '최대 연속 행수'를 반환한다.
        #   값이 클수록 굵은 정지선일 가능성이 높고, 점선·노이즈는 연속이 짧아 배제된다.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        x0, x1 = int(IMG_W * SL_X0), int(IMG_W * SL_X1)
        y0, y1 = int(IMG_H * SL_Y0), int(IMG_H * SL_Y1)
        sub = hsv[y0:y1, x0:x1]
        white = (sub[:, :, 2] >= SL_VW) & (sub[:, :, 1] <= SL_SW)
        rc = white.mean(axis=1) >= SL_COV
        best = cur = 0
        for v in rc:
            cur = cur + 1 if v else 0
            if cur > best:
                best = cur
        return best

    def update_stopline(self, img):
        # 정지선 게이트 래치(히스테리시스): 굵은 흰 띠가 보이면 ON,
        #   통과(저검출이 N프레임 연속)할 때까지 유지하다가 OFF 로 바꾼다.
        #   새 정지선에 막 진입할 때는 grace 를 해제해 다음 신호를 다시 평가한다.
        nb = self.stopline_run(img)
        self.sl_nband = nb
        was = self.sl_active
        if nb >= SL_ON:
            self.sl_active = True
            self.sl_low = 0
            if not was:
                self.grace_until = 0.0
        elif self.sl_active:
            if nb < SL_LOW:
                self.sl_low += 1
                if self.sl_low >= SL_OFF_N:
                    self.sl_active = False
                    self.sl_low = 0
            else:
                self.sl_low = 0

    def decide_go(self):
        # 신호 판단: 현재 주행 의도에 맞는 신호일 때만 GO 를 반환한다(노란색은 항상 무관).
        #   좌회전 의도면 적+초(RED_LEFT), 직진 의도면 초록 직진불(STRAIGHT)일 때만 GO.
        #   신호 없음(NONE)이면 평상 주행(GO).
        if self.signal == 'NONE':
            return True
        if self.intent == 'LEFT':
            return self.signal == 'RED_LEFT'
        return self.signal == 'STRAIGHT'

    def status(self):
        # 1초마다 진단 로그 출력(콜백/발행 수, 정지선·grace·신호·의도·속도 등)
        st = 'GO' if self.go else 'STOP'
        sl = 'SL' if self.sl_active else '--'
        g = max(0.0, self.grace_until - time.time())
        mode = '1st' if not self.first_done else 'SLgate'
        self.get_logger().info(
            'cam_cb={} pub={} {}(n{}/{}) {} grace={:.1f}s sig={} intent={} [{}] angle={:+.1f} speed={:.1f}'.format(
                self.n_cb, self.n_pub, sl, self.sl_nband, SL_ON, mode, g,
                self.signal, self.intent, st, self.angle, self.speed))

    def both_sides_yellow(self, img):
        # 바닥 양쪽 노란선 검출: 차 좌/우 노면 ROI 가 모두 노란색이면 True 를 반환한다.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YL_LO, YL_HI)
        y0, y1 = int(IMG_H * YL_Y0), int(IMG_H * YL_Y1)
        lx0, lx1 = int(IMG_W * YL_LX0), int(IMG_W * YL_LX1)
        rx0, rx1 = int(IMG_W * YL_RX0), int(IMG_W * YL_RX1)
        lc = (mask[y0:y1, lx0:lx1] > 0).mean()
        rc = (mask[y0:y1, rx0:rx1] > 0).mean()
        return lc >= YL_COV and rc >= YL_COV

    def cb(self, msg):
        # 카메라 콜백(주행 핵심): 전처리 -> 추론 -> 게이트/규칙으로 속도 결정 -> 모터 발행
        self.n_cb += 1

        # [1] 전처리: BGR -> 640x480 -> 상단크롭 -> 200x66 -> YUV -> 정규화(학습과 동일)
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        img = cv2.resize(img, (IMG_W, IMG_H))
        h = img.shape[0]
        c = img[int(h * CROP_TOP):, :]
        c = cv2.resize(c, (IN_W, IN_H))
        c = cv2.cvtColor(c, cv2.COLOR_BGR2YUV)
        x = np.transpose(c.astype(np.float32) / 255.0, (2, 0, 1))[None]

        # [2] 추론 + 역정규화: 모델 출력(정규화값)을 실제 조향/속도 단위로 복원
        with torch.no_grad():
            o = self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()[0]
        self.angle = float(o[0] * self.norm['a_s'] + self.norm['a_m'])
        self.speed = float(o[1] * self.norm['s_s'] + self.norm['s_m'])

        # [3] 진행 허가(GO/STOP) 판단: 정지선 갱신 -> 의도 판정 -> 신호등 게이트
        #   grace 중이면 무조건 GO, 첫 신호등은 신호로만, 이후엔 정지선 보일 때만 신호 적용.
        self.update_stopline(img)
        self.intent = 'LEFT' if self.angle < INTENT_LEFT_ANGLE else 'STRAIGHT'
        now = time.time()
        if now < self.grace_until:
            self.go = True
        elif not self.first_done:
            self.go = self.signal in ('STRAIGHT', 'LEFT', 'RED_LEFT')
            if self.go:
                self.first_done = True
                self.grace_until = now + GRACE_SEC
        elif self.sl_active:
            self.go = self.decide_go()
            if self.go:
                self.grace_until = now + GRACE_SEC
                if self.intent != 'LEFT':
                    self.light_pass_t = now
        else:
            self.go = True

        # [4] 모터 명령 구성: 조향은 모델값(클립), 속도는 GO 면 하한 보장, 아니면 정지
        angle_out = float(np.clip(self.angle, -MAX_ANGLE, MAX_ANGLE))
        if self.go:
            speed_out = float(max(MIN_SPEED, self.speed))
        else:
            speed_out = 0.0

        # [5] 속도 보정 규칙: (a) 신호등 통과 후 일정 구간 FIX_SPEED 고정,
        #                    (b) 바닥 양쪽 노란선이면 YL_SPEED 고정(끝나면 추론 속도 복귀)
        if self.light_pass_t is not None:
            dt = now - self.light_pass_t
            if FIX_DELAY <= dt < FIX_DELAY + FIX_DUR:
                speed_out = float(FIX_SPEED)
        self.yellow_lock = bool(self.go and self.both_sides_yellow(img))
        if self.yellow_lock:
            speed_out = float(YL_SPEED)

        # [6] 모터 명령 발행 + 표시용 프레임 저장 (속도 상한 MAX_SPEED 클립)
        m = Float32MultiArray()
        m.data = [angle_out, float(np.clip(speed_out, 0.0, MAX_SPEED))]
        self.pub_motor.publish(m)
        self.n_pub += 1
        self.vis = img

    def show(self):
        # 표시 콜백: 현재 프레임에 조향/속도·신호·정지선·노란선 상태를 그려 창에 출력
        if self.vis is None:
            return
        v = self.vis.copy()

        # 크롭 경계선 + 조향/속도 텍스트
        cv2.line(v, (0, int(IMG_H * CROP_TOP)), (IMG_W, int(IMG_H * CROP_TOP)),
                 (0, 255, 255), 1)
        cv2.putText(v, 'angle {:+.1f}  speed {:.1f}'.format(self.angle, self.speed),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 신호 + GO/STOP 상태 텍스트
        st  = 'GO ({})'.format(self.intent) if self.go else 'STOP ({})'.format(self.signal)
        col = (0, 255, 0) if self.go else (0, 0, 255)
        cv2.putText(v, 'sig:{}  {}'.format(self.signal, st),
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        # 정지선/grace/모드 상태 텍스트
        g = max(0.0, self.grace_until - time.time())
        if g > 0:
            slt = 'GRACE {:.1f}s (signal gate off)'.format(g)
            slc = (0, 255, 0)
        elif not self.first_done:
            slt = '1st light: signal-only (sig {})'.format(self.signal)
            slc = (0, 255, 255)
        elif self.sl_active:
            slt = 'STOPLINE ON (n{}/{})'.format(self.sl_nband, SL_ON)
            slc = (0, 165, 255)
        else:
            slt = 'no stopline (n{}/{})'.format(self.sl_nband, SL_ON)
            slc = (160, 160, 160)
        cv2.putText(v, slt, (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.6, slc, 2)

        # 정지선 ROI 박스 + 조향 화살표 + (노란선 고정 중이면) 표시
        roic = (0, 165, 255) if self.sl_active else (160, 160, 160)
        cv2.rectangle(v, (int(IMG_W*SL_X0), int(IMG_H*SL_Y0)),
                      (int(IMG_W*SL_X1), int(IMG_H*SL_Y1)), roic, 1)
        cx = IMG_W // 2
        cv2.arrowedLine(v, (cx, IMG_H - 20),
                        (int(cx + self.angle * 2.0), IMG_H - 70),
                        (0, 0, 255), 3, tipLength=0.3)
        if self.yellow_lock:
            cv2.putText(v, 'YELLOW LANE: speed lock {:.0f}'.format(YL_SPEED),
                        (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WIN, v)
        cv2.waitKey(1)


def main(args=None):
    # 노드 실행 진입점: 초기화 -> spin(콜백 루프) -> 종료 정리
    rclpy.init(args=args)
    node = E2EDrive()
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
