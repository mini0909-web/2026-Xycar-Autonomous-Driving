#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# e2e_sched.py - 차선주행 + 곡률 변속 스케줄링 노드 (장애물 없음)
#   Q/E 로 직진 기준 목표속도 설정 → 2차함수로 커브 시 자동 감속
#   ros2 launch track_drive sched.launch.py

import os
import csv
import time
from collections import deque
import rclpy, cv2
import numpy as np
import torch
import torch.nn as nn
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

LOG_DIR = '/home/user/ruby_ws/logs'

MODEL      = '/home/user/ruby_ws/e2e_real/model_real.pth'
NORM       = '/home/user/ruby_ws/e2e_real/norm_real.json'
IMG_W, IMG_H = 640, 480
IN_W, IN_H   = 200, 66
CROP_TOP     = 0.50
MAX_ANGLE    = 100.0
MAX_SPEED    = 30.0
BASE_SPEED   = 20.0   # 디폴트/바닥 속도 - 조향이 커지면 이 속도까지 되돌아옴
ANGLE_THRESH = 10.0   # -10~10 이내(작은 조향)면 부스트 후보, 벗어나면(큰 조향) 즉시 감속
BOOST_CONFIRM_FRAMES = 5  # -10~10이 이만큼 연속돼야 진짜 직진으로 보고 부스트 시작
                          # (S자 구간에서 각도가 순간적으로 임계값 안쪽으로 스칠 때 오인 방지)
SPEED_STEP_PER_FRAME = 1.0  # 가속/감속 전부 프레임당 이만큼씩 연속 변화
ANGLE_LAG    = 5      # 최근 N프레임 중 최댓값(peak_angle) - 로그 참고용, 속도 계산엔 미사용
SPEED_STEP   = 2.0
MIN_START    = 8.0

WIN = 'E2E Sched'


class PilotNet(nn.Module):
    def __init__(self):
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
            nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class E2ESched(Node):

    def __init__(self):
        super().__init__('e2e_sched')
        self.bridge = CvBridge()

        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(MODEL):
            self.get_logger().error('model 없음: {}'.format(MODEL))
        self.net = PilotNet().to(self.dev)
        import json
        self.net.load_state_dict(torch.load(MODEL, map_location=self.dev))
        self.net.eval()
        self.norm = json.load(open(NORM))

        self.model_angle     = 0.0
        self.target_speed    = MAX_SPEED  # Q/E로 조절하는 부스트 상한
        self.cur_speed       = 0.0
        self.adaptive_speed  = 0.0
        self.straight_streak = 0  # -20~20 조향이 연속된 프레임 수 (부스트 시작 조건)
        self.angle_buf       = deque(maxlen=ANGLE_LAG)
        self.estop          = False
        self.n_cb  = 0
        self.n_pub = 0
        self.vis   = None

        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, 'sched_{}.csv'.format(
            time.strftime('%Y%m%d_%H%M%S')))
        self._log_f = open(log_path, 'w', newline='')
        self._log_w = csv.writer(self._log_f)
        self._log_w.writerow(['timestamp', 'raw_angle', 'cmd_angle', 'peak_angle',
                               'target_speed', 'cmd_speed'])
        self.get_logger().info('로그 저장: {}'.format(log_path))

        self.pub_motor = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)
        self.create_subscription(Image, '/usb_cam/image_raw/front',
                                 self.cb, qos_profile_sensor_data)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, IMG_W, IMG_H)
        self.create_timer(0.03, self.show)
        self.get_logger().info(
            'e2e_sched ready | dev={} | S=ESTOP R=재개 Q/E=목표속도'.format(self.dev))

    def render(self, img):
        v = img.copy()
        cv2.line(v, (0, int(IMG_H * CROP_TOP)), (IMG_W, int(IMG_H * CROP_TOP)),
                 (0, 255, 255), 1)
        if self.estop:
            cv2.rectangle(v, (0, 0), (IMG_W, 50), (0, 0, 180), -1)
            cv2.putText(v, '[ ESTOP ] R 로 재개', (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(v, 'E2E SCHED', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(v, 'angle:{:+6.1f}  spd:{:.1f} -> target:{:.1f}'.format(
                    self.model_angle, self.cur_speed, self.target_speed),
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 0), 1)
        cv2.putText(v, 'cam:{}  pub:{}'.format(self.n_cb, self.n_pub),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(v, 'S:stop  R:go  Q:spd+  E:spd-',
                    (10, IMG_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cx = IMG_W // 2
        cv2.arrowedLine(v, (cx, IMG_H - 40),
                        (int(cx + self.model_angle * 2.0), IMG_H - 90),
                        (0, 0, 255), 3, tipLength=0.3)
        return v

    def cb(self, msg):
        self.n_cb += 1
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        img = cv2.resize(img, (IMG_W, IMG_H))
        c = img[int(IMG_H * CROP_TOP):, :]
        c = cv2.resize(c, (IN_W, IN_H))
        c = cv2.cvtColor(c, cv2.COLOR_BGR2YUV)
        x = np.transpose(c.astype(np.float32) / 255.0, (2, 0, 1))[None]

        with torch.no_grad():
            o = self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()[0]

        self.model_angle = float(o[0] * self.norm['a_s'] + self.norm['a_m'])

        # 조향: 매 프레임 즉시 반영 (버퍼 거치지 않음, 지연 없음)
        cmd_angle = float(np.clip(self.model_angle, -MAX_ANGLE, MAX_ANGLE))

        # 속도: 최근 ANGLE_LAG프레임 중 최댓값(peak_angle)으로 계산 → 커브는 조기 감속,
        # 직진 복귀는 버퍼가 비워질 때까지 살짝 보수적으로 (안전 쪽으로 치우침)
        self.angle_buf.append(abs(self.model_angle))
        peak_angle = max(self.angle_buf)

        # 목표 속도(desired) 결정 후, 현재 속도(cur_speed)를 매 프레임 최대
        # SPEED_STEP_PER_FRAME(=1)만큼만 그쪽으로 움직인다. ESTOP(목표 0)/재개
        # (목표 BASE_SPEED)/부스트(목표 target_speed)/큰 조향 감속(목표 BASE_SPEED)
        # 전부 이 하나의 프레임 단위 램프로 처리돼서 훅 뛰거나 훅 꺼지는 구간이 없다.
        if self.estop:
            desired = 0.0
            self.straight_streak = 0
        elif abs(self.model_angle) <= ANGLE_THRESH:
            self.straight_streak += 1
            if self.straight_streak >= BOOST_CONFIRM_FRAMES:
                desired = self.target_speed
            else:
                desired = self.cur_speed  # 아직 직진 확정 전 - 현재 속도 유지
        else:
            self.straight_streak = 0
            desired = BASE_SPEED

        if self.cur_speed < desired:
            self.cur_speed = min(self.cur_speed + SPEED_STEP_PER_FRAME, desired)
        elif self.cur_speed > desired:
            self.cur_speed = max(self.cur_speed - SPEED_STEP_PER_FRAME, desired)
        self.adaptive_speed = self.cur_speed

        m = Float32MultiArray()
        m.data = [cmd_angle, float(self.cur_speed)]
        self.pub_motor.publish(m)
        self.n_pub += 1
        self.vis = self.render(img)

        self._log_w.writerow([
            '{:.4f}'.format(time.time()),
            '{:.4f}'.format(self.model_angle),
            '{:.4f}'.format(cmd_angle),
            '{:.4f}'.format(peak_angle),
            '{:.4f}'.format(self.target_speed),
            '{:.4f}'.format(self.cur_speed),
        ])

    def show(self):
        if self.vis is None:
            return
        cv2.imshow(WIN, self.vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            self.estop = True
            self.get_logger().warn('ESTOP! (프레임 단위로 0까지 감속)')
        elif k == ord('r'):
            self.estop = False
            self.get_logger().info('재개 (프레임 단위로 {:.1f}까지 가속)'.format(BASE_SPEED))
        # Q/E 수동 속도 조절은 일단 비활성화 (target_speed는 MAX_SPEED 고정)


def main(args=None):
    rclpy.init(args=args)
    node = E2ESched()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        m = Float32MultiArray()
        m.data = [0.0, 0.0]
        node.pub_motor.publish(m)
        node._log_f.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
