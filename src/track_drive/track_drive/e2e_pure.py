#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# e2e_pure.py - 카메라 → PilotNet → angle 추론 후 /motor_lane 발행
#   속도/ESTOP 제어는 mux_node로 위임

import os
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

# CPU 추론 시 PyTorch 기본값(전체 코어 인트라옵 병렬화)은 이렇게 작은 모델/입력엔
# 스레드 생성/동기화 오버헤드가 실제 연산량보다 커서 오히려 CPU를 더 먹고 더 느림.
# 매 프레임(카메라 fps)마다 반복 호출되므로 싱글스레드로 고정. cv2도 같은 이유로 고정.
torch.set_num_threads(1)
cv2.setNumThreads(1)

MODEL    = '/home/user/ruby_ws/e2e_real/model_real.pth'
NORM     = '/home/user/ruby_ws/e2e_real/norm_real.json'
IMG_W, IMG_H = 640, 480
IN_W, IN_H   = 200, 66
CROP_TOP     = 0.50
MAX_ANGLE    = 100.0
ANGLE_LAG    = 5   # 최근 N프레임 중 최댓값(peak_angle)을 motor_lane[1]에 실어 보냄 (속도용, 조향엔 영향 없음)

WIN = 'E2E Cam'


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


class E2EPure(Node):

    def __init__(self):
        super().__init__('e2e_pure')
        self.bridge = CvBridge()

        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(MODEL):
            self.get_logger().error('model 없음: {}'.format(MODEL))
        self.net = PilotNet().to(self.dev)
        import json
        self.net.load_state_dict(torch.load(MODEL, map_location=self.dev))
        self.net.eval()
        self.norm = json.load(open(NORM))

        self.model_angle = 0.0
        self.angle_buf   = deque(maxlen=ANGLE_LAG)
        self.n_cb  = 0
        self.n_pub = 0
        self.vis   = None

        self.pub_motor = self.create_publisher(Float32MultiArray, '/motor_lane', 10)
        self.create_subscription(Image, '/usb_cam/image_raw/front',
                                 self.cb, qos_profile_sensor_data)
        # 시각화 꺼둠(원격 화면 렉 때문에 통합런치 경량화)
        # cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(WIN, IMG_W, IMG_H)
        # self.create_timer(0.03, self.show)
        self.get_logger().info('e2e_pure ready | dev={}'.format(self.dev))

    def render(self, img):
        v = img.copy()
        cv2.line(v, (0, int(IMG_H * CROP_TOP)), (IMG_W, int(IMG_H * CROP_TOP)),
                 (0, 255, 255), 1)
        cv2.putText(v, 'angle:{:+6.1f}'.format(self.model_angle),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(v, 'cam:{} pub:{}'.format(self.n_cb, self.n_pub),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cx = IMG_W // 2
        cv2.arrowedLine(v, (cx, IMG_H - 40),
                        (int(cx + self.model_angle * 2.0), IMG_H - 90),
                        (0, 0, 255), 3, tipLength=0.3)
        return v

    def cb(self, msg):
        self.n_cb += 1
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        img = cv2.resize(img, (IMG_W, IMG_H))
        h = img.shape[0]
        c = img[int(h * CROP_TOP):, :]
        c = cv2.resize(c, (IN_W, IN_H))
        c = cv2.cvtColor(c, cv2.COLOR_BGR2YUV)
        x = np.transpose(c.astype(np.float32) / 255.0, (2, 0, 1))[None]

        with torch.no_grad():
            o = self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()[0]

        self.model_angle = float(o[0] * self.norm['a_s'] + self.norm['a_m'])

        # 조향용 각도는 매 프레임 즉시값. peak_angle(속도용)은 최근 ANGLE_LAG프레임 중 최댓값.
        self.angle_buf.append(abs(self.model_angle))
        peak_angle = max(self.angle_buf)

        m = Float32MultiArray()
        m.data = [float(np.clip(self.model_angle, -MAX_ANGLE, MAX_ANGLE)), peak_angle]
        self.pub_motor.publish(m)
        self.n_pub += 1

    def show(self):
        if self.vis is None:
            return
        cv2.imshow(WIN, self.vis)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = E2EPure()
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
