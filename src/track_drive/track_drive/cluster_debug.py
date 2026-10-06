#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_debug.py - 라이다 클러스터 폭/거리/포인트수 실시간 확인용 디버그 노드
  모터 제어 없음(순수 시각화). 콘 vs 장애물 클러스터 폭 임계값을 실측으로 잡기 위한 도구.
  실제 콘/장애물을 라이다 앞에 놓고 화면에 찍히는 width_cm 값을 읽으면 됨.
"""
import math
import rclpy, cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

# 이 화면 좌표 규칙 기준으로 실측 확인된 보정값
DEFAULTS = {
    'YAW_OFF': 270,   # 실제 오프셋 = 트랙바값-180
    'MIRROR':    0,
    'MIN_CM':   15,   # 유효 최소 거리(cm)
    'MAX_CM':  600,   # 유효 최대 거리(cm)
    'GAP_CM':   20,   # 이 이상 벌어지면 다른 클러스터로 분리 (cm)
    'MIN_PTS':   3,   # 클러스터 최소 포인트 수
    # ── 사각형 ROI (이 안쪽만 클러스터링 대상) ────────────
    'ROI_X_CM':  60,  # x 반폭(cm) - |x| <= 이 값 (실측으로 -60~60 확인됨)
    'ROI_Y_CM':  50,  # y 반폭(cm) - |y| <= 이 값
}
TRACKBAR_MAX = {
    'YAW_OFF': 360, 'MIRROR': 1, 'MIN_CM': 200, 'MAX_CM': 1000,
    'GAP_CM': 100, 'MIN_PTS': 20, 'ROI_X_CM': 500, 'ROI_Y_CM': 300,
}

VIEW = 1.0   # 전방 1m, 좌우 ±1m
SCL  = 300
VH   = int(VIEW * SCL)
VW   = VH * 2
CPX  = (VW // 2, VH - 5)
PARAMS_WIN = 'Params'
VIEW_WIN   = 'Cluster Debug'

COLORS = [
    (30, 130, 255), (0, 200, 100), (0, 165, 255), (255, 100, 100),
    (200, 0, 200), (0, 220, 220), (100, 180, 0), (180, 100, 255),
]


def m2px(x, y):
    return int(VW // 2 + x * SCL), int(VH - y * SCL)


def _nop(_):
    pass


class ClusterDebugNode(Node):

    def __init__(self):
        super().__init__('cluster_debug')
        self.create_subscription(LaserScan, '/scan', self.cb, qos_profile_sensor_data)

        cv2.namedWindow(PARAMS_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PARAMS_WIN, 360, 280)
        for name, val in DEFAULTS.items():
            cv2.createTrackbar(name, PARAMS_WIN, val, TRACKBAR_MAX[name], _nop)

        self.get_logger().info('클러스터 디버그 준비됨 (모터 제어 없음). 콘/장애물 앞에 놓고 width_cm 읽을 것')

    def _read_params(self):
        g = lambda n: cv2.getTrackbarPos(n, PARAMS_WIN)
        return {
            'YAW_OFFSET': float(g('YAW_OFF') - 180),
            'MIRROR': bool(g('MIRROR')),
            'MIN_DIST': g('MIN_CM') / 100.0,
            'MAX_DIST': g('MAX_CM') / 100.0,
            'GAP': g('GAP_CM') / 100.0,
            'MIN_PTS': max(1, g('MIN_PTS')),
            'ROI_X': g('ROI_X_CM') / 100.0,
            'ROI_Y': g('ROI_Y_CM') / 100.0,
        }

    def _logical_deg(self, n, p):
        raw_deg = np.arange(n, dtype=np.float32) * 360.0 / n
        if p['MIRROR']:
            raw_deg = (-raw_deg) % 360.0
        return (raw_deg + p['YAW_OFFSET']) % 360.0

    def _cluster(self, xs, ys, gap):
        """(x,y) 포인트를 유클리드 간격 기준으로 클러스터링"""
        clusters, current = [], []
        for x, y in zip(xs, ys):
            if current and math.hypot(x - current[-1][0], y - current[-1][1]) > gap:
                clusters.append(current)
                current = []
            current.append((x, y))
        if current:
            clusters.append(current)
        return clusters

    def cb(self, msg):
        p = self._read_params()
        r = np.array(msg.ranges, dtype=np.float32)
        n = len(r)
        logical_deg = self._logical_deg(n, p)

        ok = np.isfinite(r) & (r > p['MIN_DIST']) & (r < p['MAX_DIST'])
        idx = np.where(ok)[0]
        th = np.radians(logical_deg[idx])
        xs = r[idx] * np.cos(th)
        ys = -r[idx] * np.sin(th)

        # 사각형 ROI: |x| <= ROI_X, |y| <= ROI_Y 안쪽만 클러스터링 대상으로
        roi = (np.abs(xs) <= p['ROI_X']) & (np.abs(ys) <= p['ROI_Y'])
        xs, ys = xs[roi].tolist(), ys[roi].tolist()

        clusters = self._cluster(xs, ys, p['GAP'])
        clusters = [c for c in clusters if len(c) >= p['MIN_PTS']]

        self.draw(clusters, p)

    def draw(self, clusters, p):
        img = np.full((VH, VW, 3), 245, np.uint8)

        for d in range(1, int(VIEW) + 1):
            row = VH - int(d * SCL)
            if 0 <= row < VH:
                cv2.line(img, (0, row), (VW, row), (215, 215, 215), 1)
                cv2.putText(img, f'{d}m', (2, row - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (175, 175, 175), 1)
        cv2.line(img, (VW // 2, 0), (VW // 2, VH), (215, 215, 215), 1)

        # ROI 사각형 (|x| <= ROI_X, |y| <= ROI_Y)
        p1 = m2px(-p['ROI_X'], -p['ROI_Y'])
        p2 = m2px(p['ROI_X'], p['ROI_Y'])
        cv2.rectangle(img, p1, p2, (0, 130, 0), 1)

        cv2.circle(img, CPX, 6, (0, 0, 180), -1)

        for i, cluster in enumerate(clusters):
            col = COLORS[i % len(COLORS)]
            cxs = [p[0] for p in cluster]
            cys = [p[1] for p in cluster]
            width_cm = (max(cys) - min(cys)) * 100.0
            dist_m = min(math.hypot(x, y) for x, y in cluster)
            cx_m, cy_m = sum(cxs) / len(cxs), sum(cys) / len(cys)

            for x, y in cluster:
                cv2.circle(img, m2px(x, y), 3, col, -1)

            label_pos = m2px(cx_m, cy_m)
            label = f'{width_cm:.0f}cm {dist_m:.2f}m n{len(cluster)}'
            cv2.putText(img, label, (label_pos[0] + 6, label_pos[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
            cv2.putText(img, label, (label_pos[0] + 5, label_pos[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        cv2.putText(img, f'clusters: {len(clusters)}', (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 1)
        cv2.putText(img, 'motor control none (debug only)', (5, VH - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        cv2.imshow(VIEW_WIN, img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ClusterDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
