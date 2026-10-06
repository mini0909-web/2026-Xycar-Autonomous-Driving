#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# e2e_pure_clahe.py - CLAHE 전처리 적용 버전 순수 E2E 주행 노드
#   e2e_pure.py 와 동일하되 YUV 변환 후 Y채널에 CLAHE 적용 (반사광 완화).
#   model_clahe.pth + norm_clahe.json 을 사용한다.
#   S키: ESTOP(즉시 정지), R키: 재개
# ============================================================================

import os, time, datetime
import rclpy, cv2
import numpy as np
import torch
import torch.nn as nn
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

MODEL = '/home/user/ruby_ws/e2e_clahe/model_clahe.pth'
NORM  = '/home/user/ruby_ws/e2e_clahe/norm_clahe.json'
IMG_W, IMG_H = 640, 480
IN_W, IN_H   = 200, 66
CROP_TOP     = 0.40
MAX_ANGLE    = 100.0
MIN_SPEED    = 10.0
MAX_SPEED    = 20.0
SPEED_RAMP   = 1.0   # 프레임당 최대 속도 증가량 (급가속 방지)

REC_DIR = os.environ.get('E2E_REC_DIR', '/home/user/ruby_ws/e2e_clahe/rec')
REC_FPS = float(os.environ.get('E2E_REC_FPS', '10'))
WIN = 'E2E Pure CLAHE'

CLAHE_OBJ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


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
            nn.Linear(10, 2),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class E2EPureClahe(Node):

    def __init__(self):
        super().__init__('e2e_pure_clahe')
        self.bridge = CvBridge()

        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(MODEL):
            self.get_logger().error('no model: {} (train_clahe.py 먼저)'.format(MODEL))
        self.net = PilotNet().to(self.dev)
        import json
        self.net.load_state_dict(torch.load(MODEL, map_location=self.dev))
        self.net.eval()
        self.norm = json.load(open(NORM))

        self.angle = 0.0
        self.speed = 0.0
        self.vis = None
        self.n_cb = 0
        self.n_pub = 0
        self.n_rec = 0
        self.estop = False
        self.cur_speed = 0.0

        self.writer = None

        self.pub_motor = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)
        self.create_subscription(Image, '/usb_cam/image_raw/front',
                                 self.cb, qos_profile_sensor_data)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, IMG_W, IMG_H)
        self.timer = self.create_timer(0.03, self.show)
        self.diag  = self.create_timer(1.0, self.status)
        self.get_logger().info('e2e_pure_clahe ready dev={}'.format(self.dev))

    def status(self):
        self.get_logger().info(
            'cam_cb={} pub={} rec={} angle={:+.1f} speed={:.1f} estop={}'.format(
                self.n_cb, self.n_pub, self.n_rec, self.angle, self.speed, self.estop))

    def render(self, img):
        v = img.copy()
        cv2.line(v, (0, int(IMG_H * CROP_TOP)), (IMG_W, int(IMG_H * CROP_TOP)),
                 (0, 255, 255), 1)
        label = 'CLAHE E2E  [ESTOP]' if self.estop else 'CLAHE E2E'
        color = (0, 0, 255) if self.estop else (0, 255, 0)
        cv2.putText(v, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(v, 'angle {:+.1f}  speed {:.1f}'.format(self.angle, self.speed),
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cx = IMG_W // 2
        cv2.arrowedLine(v, (cx, IMG_H - 20),
                        (int(cx + self.angle * 2.0), IMG_H - 70),
                        (0, 0, 255), 3, tipLength=0.3)
        if self.writer is not None:
            cv2.circle(v, (IMG_W - 30, 30), 8, (0, 0, 255), -1)
            cv2.putText(v, 'REC', (IMG_W - 80, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return v

    def cb(self, msg):
        self.n_cb += 1

        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        img = cv2.resize(img, (IMG_W, IMG_H))
        h = img.shape[0]
        c = img[int(h * CROP_TOP):, :]
        c = cv2.resize(c, (IN_W, IN_H))
        c = cv2.cvtColor(c, cv2.COLOR_BGR2YUV)
        c[:, :, 0] = CLAHE_OBJ.apply(c[:, :, 0])  # Y채널 CLAHE
        x = np.transpose(c.astype(np.float32) / 255.0, (2, 0, 1))[None]

        with torch.no_grad():
            o = self.net(torch.from_numpy(x).to(self.dev)).cpu().numpy()[0]
        self.angle = float(o[0] * self.norm['a_s'] + self.norm['a_m'])
        self.speed = float(o[1] * self.norm['s_s'] + self.norm['s_m'])

        m = Float32MultiArray()
        if self.estop:
            self.cur_speed = 0.0
            m.data = [0.0, 0.0]
        else:
            target = float(np.clip(max(MIN_SPEED, self.speed), 0.0, MAX_SPEED))
            self.cur_speed = min(self.cur_speed + SPEED_RAMP, target)
            m.data = [float(np.clip(self.angle, -MAX_ANGLE, MAX_ANGLE)), self.cur_speed]
        self.pub_motor.publish(m)
        self.n_pub += 1

        v = self.render(img)
        self.vis = v
        if self.writer is not None:
            self.writer.write(v)
            self.n_rec += 1

    def show(self):
        if self.vis is None:
            return
        cv2.imshow(WIN, self.vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            self.estop = True
            self.get_logger().warn('ESTOP! (r 로 재개)')
        elif k == ord('r'):
            self.estop = False
            self.cur_speed = MIN_SPEED
            self.get_logger().info('재개')

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info('saved video ({} frames) -> {}'.format(
                self.n_rec, self.rec_path))
            self.writer = None


def main(args=None):
    rclpy.init(args=args)
    node = E2EPureClahe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
