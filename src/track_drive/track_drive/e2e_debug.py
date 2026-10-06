#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# e2e_debug.py - 차선주행 디버그 노드 (단독 실행용)
#   /xycar_motor 직접 발행, S/R/Q/E 키로 ESTOP/속도 제어
#   ros2 launch track_drive pure.launch.py

import os
import csv
import time
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
SPEED_STEP   = 2.0
SPEED_RAMP   = 2.0
MIN_START    = 8.0

WIN = 'E2E Debug'


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


class E2EDebug(Node):

    def __init__(self):
        super().__init__('e2e_debug')
        self.bridge = CvBridge()

        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(MODEL):
            self.get_logger().error('model 없음: {}'.format(MODEL))
        self.net = PilotNet().to(self.dev)
        import json
        self.net.load_state_dict(torch.load(MODEL, map_location=self.dev))
        self.net.eval()
        self.norm = json.load(open(NORM))

        self.model_angle  = 0.0
        self.target_speed = 0.0
        self.cur_speed    = 0.0
        self.estop        = False
        self.n_cb  = 0
        self.n_pub = 0
        self.vis   = None

        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, 'drive_{}.csv'.format(
            time.strftime('%Y%m%d_%H%M%S')))
        self._log_f = open(log_path, 'w', newline='')
        self._log_w = csv.writer(self._log_f)
        self._log_w.writerow(['timestamp', 'raw_angle', 'cmd_angle', 'cmd_speed'])
        self.get_logger().info('로그 저장: {}'.format(log_path))

        self.pub_motor = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)
        self.create_subscription(Image, '/usb_cam/image_raw/front',
                                 self.cb, qos_profile_sensor_data)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, IMG_W, IMG_H)
        self.create_timer(0.03, self.show)
        self.get_logger().info('e2e_debug ready | dev={} | S=ESTOP R=재개 Q/E=속도'.format(self.dev))

    def render(self, img):
        v = img.copy()
        cv2.line(v, (0, int(IMG_H * CROP_TOP)), (IMG_W, int(IMG_H * CROP_TOP)),
                 (0, 255, 255), 1)
        if self.estop:
            cv2.rectangle(v, (0, 0), (IMG_W, 50), (0, 0, 180), -1)
            cv2.putText(v, '[ ESTOP ] R 로 재개', (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(v, 'E2E DEBUG', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(v, 'angle:{:+6.1f}  spd:{:.1f} -> target:{:.1f}'.format(
                    self.model_angle, self.cur_speed, self.target_speed),
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
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

        m = Float32MultiArray()
        if self.estop:
            self.cur_speed = 0.0
            m.data = [0.0, 0.0]
        else:
            if self.cur_speed < self.target_speed:
                self.cur_speed = min(self.cur_speed + SPEED_RAMP, self.target_speed)
            else:
                self.cur_speed = self.target_speed
            m.data = [float(np.clip(self.model_angle, -MAX_ANGLE, MAX_ANGLE)),
                      float(self.cur_speed)]
        self.pub_motor.publish(m)
        self.n_pub += 1
        self.vis = self.render(img)

        cmd_angle = float(np.clip(self.model_angle, -MAX_ANGLE, MAX_ANGLE))
        self._log_w.writerow([
            '{:.4f}'.format(time.time()),
            '{:.4f}'.format(self.model_angle),
            '{:.4f}'.format(cmd_angle),
            '{:.4f}'.format(self.cur_speed),
        ])

    def show(self):
        if self.vis is None:
            return
        cv2.imshow(WIN, self.vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            self.estop = True
            self.cur_speed = 0.0
            self.get_logger().warn('ESTOP!')
        elif k == ord('r'):
            self.estop = False
            self.cur_speed = min(MIN_START, self.target_speed) if self.target_speed > 0 else 0.0
            self.get_logger().info('재개 (target={:.1f})'.format(self.target_speed))
        elif k == ord('q'):
            self.target_speed = min(self.target_speed + SPEED_STEP, MAX_SPEED)
            self.get_logger().info('speed+ target={:.1f}'.format(self.target_speed))
        elif k == ord('e'):
            self.target_speed = max(self.target_speed - SPEED_STEP, 0.0)
            self.get_logger().info('speed- target={:.1f}'.format(self.target_speed))


def main(args=None):
    rclpy.init(args=args)
    node = E2EDebug()
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
