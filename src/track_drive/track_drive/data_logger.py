#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# data_logger.py - 단일 전방 카메라 E2E 데이터 수집 노드
#   전방 프레임마다 (이미지 + 그 순간의 수동 angle/speed)를 한 쌍으로 저장한다.
#   labels.csv 컬럼: frame,image,angle,speed,t  -> train.py 가 그대로 학습에 사용.
#   조작: r/SPACE = 기록 토글, q/ESC = 종료. 화면의 OK/NO CMD 로 수동명령 수신 확인.
#   * 수집 중에는 자율주행을 끄고 teleop_key + data_logger 만 실행한다. *
# ============================================================================

import os, csv, time
import rclpy, cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

# 입력 토픽 / 저장 루트 / 해상도 / 창 이름
CAM_F = '/usb_cam/image_raw/front'
OUT_ROOT = '/home/user/ruby_ws/dataset_real'
IMG_W, IMG_H = 640, 480
WIN = 'DataLogger (r=rec toggle, q=quit)'


# 데이터 수집 노드: 카메라 프레임과 수동 조향·속도를 받아 세션 폴더에 저장한다.
class DataLogger(Node):

    def __init__(self):
        # 노드 초기화: 상태 변수 준비, 세션 폴더/CSV 생성, 카메라·수동명령 구독, 타이머 설정
        super().__init__('data_logger')
        self.bridge = CvBridge()
        self.f = None
        self.angle = 0.0
        self.speed = 0.0
        self.last_cmd_t = 0.0
        self.recording = True
        self.count = 0

        # 세션별 저장 폴더(images/)와 labels.csv 준비
        ses = time.strftime('%Y%m%d_%H%M%S')
        root = os.path.join(OUT_ROOT, ses)
        self.d_f = os.path.join(root, 'images')
        os.makedirs(self.d_f, exist_ok=True)
        self.csv_path = os.path.join(root, 'labels.csv')
        self.csv = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv)
        self.writer.writerow(['frame', 'image', 'angle', 'speed', 't'])

        # 카메라 구독 + 수동명령 구독(namespace 유무 둘 다) + 미리보기 타이머
        self.create_subscription(Image, CAM_F, self.cb_f, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, '/xycar_motor', self.cb_cmd, 10)
        self.timer = self.create_timer(1.0, self.tick)
        self.get_logger().info('data_logger (camera only) ready -> {}'.format(self.csv_path))

    def cb_cmd(self, m):
        # 수동명령 콜백: 최신 조향·속도와 수신 시각을 기록(라벨로 저장될 값)
        self.angle = float(m.data[0])
        self.speed = float(m.data[1])
        self.last_cmd_t = time.time()

    def cb_f(self, msg):
        # 카메라 콜백(저장 핵심): 프레임을 보관하고, 기록 중이면 이미지+라벨 한 줄을 저장
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.f = cv2.resize(img, (IMG_W, IMG_H))
        if not self.recording:
            return
        name = '{:06d}.jpg'.format(self.count)
        cv2.imwrite(os.path.join(self.d_f, name), self.f)
        self.writer.writerow([self.count, name, '{:.4f}'.format(self.angle),
                              '{:.4f}'.format(self.speed), '{:.3f}'.format(time.time())])
        self.csv.flush()
        self.count += 1

    def tick(self):
        # 1초마다 상태 로그 출력 (cv2 창 없이 터미널로만 표시)
        cmd_ok = (time.time() - self.last_cmd_t) < 0.5
        self.get_logger().info(
            'recording={} frames={} a={:.1f} s={:.1f} cmd={}'.format(
                self.recording, self.count,
                self.angle, self.speed,
                'OK' if cmd_ok else 'NO CMD'))


def main(args=None):
    # 노드 실행 진입점: 초기화 -> spin -> 종료 시 CSV 안전 종료/정리
    rclpy.init(args=args)
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.csv.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
