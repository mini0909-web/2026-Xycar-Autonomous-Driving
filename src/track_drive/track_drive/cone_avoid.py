#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cone_avoid.py - 라이다 좌우 섹터 P제어
  - 좌측 섹터 거리 vs 우측 섹터 거리 차이로 조향
  - angle = KP * (right_dist - left_dist)
  - 파라미터는 'Params' 창 트랙바로 실시간 조정 (콘 잡을 때까지 튜닝용)
  - YAW_OFF/MIRROR: 라이다 장착 방향이 안 맞을 때 보정용
    (알려진 위치에 물건 두고, 화면에서 실제 방향의 색(파랑=왼쪽/초록=오른쪽)에
     찍히도록 두 트랙바를 조절해서 찾을 것)
"""
import math
import time
import rclpy, cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Bool
from rclpy.qos import qos_profile_sensor_data

# 작은 이미지 연산을 매 스캔마다 멀티스레드로 돌리면 스레드 생성/동기화 오버헤드가
# 더 커서 오히려 CPU를 더 먹음 (e2e_pure의 torch.set_num_threads(1)과 같은 이유).
cv2.setNumThreads(1)

# ── 주행 파라미터 ─────────────────────────────────────────
CONE_SPEED = 12.5
MAX_ANGLE  = 100

# ── 트랙바 초기값 ─────────────────────────────────────────
# 0=전방 반시계: 90=왼쪽, 270=오른쪽 (보정 후 기준)
DEFAULTS = {
    'L_DEG1':   15,   # 왼쪽 섹터 시작(도)
    'L_DEG2':   80,   # 왼쪽 섹터 끝(도)
    'R_DEG1':  280,   # 오른쪽 섹터 시작(도)
    'R_DEG2':  345,   # 오른쪽 섹터 끝(도)
    'MIN_CM':   15,   # 유효 최소 거리(cm)
    'MAX_CM':  600,   # 유효 최대 거리(cm)
    'KP':       75,   # P 게인(정수, 원래 75.5)
    'EMA_PCT':  95,   # EMA 계수 * 100 (낮을수록 부드럽)
    'YAW_OFF':   0,   # 라이다 회전 보정(도), 180=오프셋 0 (실제값 = 트랙바값-180). 실차 확인 결과 180도 뒤집혀 있어서 0으로 고정
    'MIRROR':    0,   # 라이다가 뒤집혀 좌우/회전방향이 반대면 1
    # ── 콘 게이트(모드 전환 트리거) ─────────────────────────
    'GAP_CM':   20,   # 클러스터 분리 간격(cm)
    'CLU_PTS':   3,   # 클러스터 최소 포인트 수
    # cluster_debug.py에서 실측한 ROI (좌우 ±60cm, 전방 ±300cm)를 그대로 반영
    'ROI_X_CM': 65,   # 좌우 반폭(cm) - |x| <= 이 값
    'ROI_Y_CM': 300,  # 전방 반폭(cm) - |y| <= 이 값
    'CONE_MIN_CM': 3,   # 이 폭(cm) 이상
    'CONE_CM':  12,   # 이 폭(cm) 이하인 클러스터만 콘으로 인정 (범위: MIN~CONE_CM)
    'GATE_N':    2,   # 좌/우 각각 이 개수 이상이어야 게이트 인정 (좌우 합쳐 총 GATE_N*2개)
    'CONFIRM':   1,   # 좌우 동시 확인 필요 연속 스캔 수
    'EXIT_MS':   10,   # 이탈 유지시간(x100ms), 10=1.0초
    # ── 동적 ROI (커브 중 진행방향 예측 전단(shear)) ──────────
    # mux 최종출력 /xycar_motor 기반 - 지금 실제로 차가 받는 명령 그대로라
    # cone_active 여부를 따로 구분할 필요 없음.
    'ANGLE_SCALE_PCT': 20, # raw angle(-100~100) -> 실제 조향각(도) 환산 비율. 실측: raw 100 ~= 실제 20도
    'ROT_GAIN_PCT': 100,   # 환산된 실제각도 대비 ROI 기울임 비율(%)
    'ROT_MAX_DEG':  30,    # ROI 기울임각 상한(도, 환산 후 기준)
    'MOTOR_TIMEOUT_MS': 5, # /xycar_motor 최신값 유효시간(x100ms), 5=0.5초. 끊기면 기울임 0으로
    'MOTOR_EMA_PCT': 30,   # 스무딩(%). 낮을수록 순간 오실레이션을 더 눌러줌(진짜 커브는 여전히 따라감)
}
TRACKBAR_MAX = {
    'L_DEG1': 180, 'L_DEG2': 180, 'R_DEG1': 360, 'R_DEG2': 360,
    'MIN_CM': 200, 'MAX_CM': 1000, 'KP': 300, 'EMA_PCT': 100,
    'YAW_OFF': 360, 'MIRROR': 1,
    'GAP_CM': 100, 'CLU_PTS': 20, 'ROI_X_CM': 300, 'ROI_Y_CM': 500,
    'CONE_MIN_CM': 100, 'CONE_CM': 100, 'GATE_N': 10, 'CONFIRM': 20, 'EXIT_MS': 50,
    'ANGLE_SCALE_PCT': 100, 'ROT_GAIN_PCT': 200, 'ROT_MAX_DEG': 90, 'MOTOR_TIMEOUT_MS': 50,
    'MOTOR_EMA_PCT': 100,
}

# 트랙바 창 자체를 안 띄우고 DEFAULTS 고정값을 씀 (CPU/창 오버헤드 절약).
# 국민대 등에서 튜닝할 때만 이 노드에 한해 True로 바꿔서 씀.
ENABLE_TRACKBAR = False

# ── 시각화 ───────────────────────────────────────────────
VIEW = 1.0   # 전방 1m, 좌우 ±1m
SCL  = 300
VH   = int(VIEW * SCL)
VW   = VH * 2
CPX  = (VW // 2, VH - 5)
PARAMS_WIN = 'Params'
VIEW_WIN   = 'Cone Avoid'

def m2px(x, y):
    return int(VW // 2 + x * SCL), int(VH - y * SCL)


def _nop(_):
    pass


class ConeAvoidNode(Node):

    def __init__(self):
        super().__init__('cone_avoid')
        # mux가 /xycar_motor의 유일한 발행자. 이 노드는 /cmd/cone + /cone/active만 내보냄
        # (active일 때만 /cmd/cone 발행 - ESTOP은 mux가 전담)
        self.pub = self.create_publisher(Float32MultiArray, '/cmd/cone', 10)
        self.active_pub = self.create_publisher(Bool, '/cone/active', 10)
        self.create_subscription(
            LaserScan, '/scan', self.cb, qos_profile_sensor_data)
        # 동적 ROI 회전용: mux가 지금 실제로 내보내는 최종 명령 각도
        self.create_subscription(
            Float32MultiArray, '/xycar_motor', self.cb_motor, 10)
        self.motor_angle = 0.0
        self.motor_angle_at = None
        self.prev_angle = 0.0

        # 콘 게이트(좌우 동시 확인) 상태
        self.gate_confirm_count = 0
        self.gate_clear_since = None
        self.cone_active = False

        if ENABLE_TRACKBAR:
            cv2.namedWindow(PARAMS_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PARAMS_WIN, 360, 480)
            for name, val in DEFAULTS.items():
                cv2.createTrackbar(name, PARAMS_WIN, val, TRACKBAR_MAX[name], _nop)

        # 시각화 꺼둠(원격 화면 렉 때문에 통합런치 경량화) - 렌더 타이머 자체를
        # 안 만들어서 draw()가 절대 호출되지 않는다. self._viz 대입은 스캔 콜백에서
        # 원래 계산되는 값 저장일 뿐이라 그대로 둬도 비용 거의 없음.
        self._viz = None
        # self.create_timer(0.1, self._render_timer)

        self.get_logger().info('준비됨. Params 창 트랙바로 튜닝 (ESTOP은 mux에서 S/R로)')

    @staticmethod
    def _g(name):
        return (cv2.getTrackbarPos(name, PARAMS_WIN) if ENABLE_TRACKBAR else DEFAULTS[name])

    def cb_motor(self, msg):
        """mux 최종출력 /xycar_motor 각도를 EMA로 스무딩해서 저장 - 순간 오실레이션이
        ROI 전단에 그대로 반영되지 않게 함 (진짜 커브는 몇 프레임 안에 따라감)."""
        if len(msg.data) >= 1:
            raw = float(msg.data[0])
            alpha = self._g('MOTOR_EMA_PCT') / 100.0
            if self.motor_angle_at is None:
                self.motor_angle = raw
            else:
                self.motor_angle = alpha * raw + (1.0 - alpha) * self.motor_angle
            self.motor_angle_at = time.monotonic()

    # ── 트랙바 값 읽기 (꺼져있으면 DEFAULTS 고정값) ─────────
    def _read_params(self):
        g = (lambda n: cv2.getTrackbarPos(n, PARAMS_WIN)) if ENABLE_TRACKBAR else (lambda n: DEFAULTS[n])
        return {
            'L_DEG1': g('L_DEG1'), 'L_DEG2': g('L_DEG2'),
            'R_DEG1': g('R_DEG1'), 'R_DEG2': g('R_DEG2'),
            'MIN_DIST': g('MIN_CM') / 100.0,
            'MAX_DIST': g('MAX_CM') / 100.0,
            'KP': float(g('KP')),
            'EMA': g('EMA_PCT') / 100.0,
            'YAW_OFFSET': float(g('YAW_OFF') - 180),
            'MIRROR': bool(g('MIRROR')),
            'GAP': g('GAP_CM') / 100.0,
            'CLU_PTS': max(1, g('CLU_PTS')),
            'ROI_X': g('ROI_X_CM') / 100.0,
            'ROI_Y': g('ROI_Y_CM') / 100.0,
            'CONE_W_MIN': g('CONE_MIN_CM') / 100.0,
            'CONE_W': g('CONE_CM') / 100.0,
            'GATE_N': max(1, g('GATE_N')),
            'CONFIRM': max(1, g('CONFIRM')),
            'EXIT_SEC': g('EXIT_MS') / 10.0,
            'ANGLE_SCALE': g('ANGLE_SCALE_PCT') / 100.0,
            'ROT_GAIN': g('ROT_GAIN_PCT') / 100.0,
            'ROT_MAX': float(g('ROT_MAX_DEG')),
            'MOTOR_TIMEOUT': g('MOTOR_TIMEOUT_MS') / 10.0,
        }

    # ── 라이다 방향 보정 ──────────────────────────────────
    def _logical_deg(self, n, p):
        """각 인덱스의 '보정된' 각도(도). MIRROR로 회전방향 반전, YAW_OFFSET으로 기준 회전."""
        raw_deg = np.arange(n, dtype=np.float32) * 360.0 / n
        if p['MIRROR']:
            raw_deg = (-raw_deg) % 360.0
        return (raw_deg + p['YAW_OFFSET']) % 360.0

    # ── 섹터 거리 ─────────────────────────────────────────
    def sector_dist(self, ranges, logical_deg, d1, d2, min_dist, max_dist):
        """섹터 내 가장 가까운 30% 평균 거리 (없으면 max_dist)"""
        mask = ((logical_deg >= d1) & (logical_deg <= d2)
                & np.isfinite(ranges) & (ranges > min_dist) & (ranges < max_dist))
        ok = ranges[mask]
        if len(ok) == 0:
            return max_dist
        k = max(1, len(ok) // 3)
        return float(np.mean(np.sort(ok)[:k]))

    # ── 진행방향 예측 각도 (mux 최종출력 /xycar_motor 기반) ──
    def _path_heading_rad(self, p):
        """ROI를 지금 진행방향으로 같이 기울이기 위한 각도. mux가 지금 실제로
        내보내는 명령 각도(raw 단위)를 그대로 씀 - cone_active 여부 구분 불필요.
        raw 단위(-100~100)는 ANGLE_SCALE로 실제 조향각(도)으로 환산 후 ROT_GAIN/ROT_MAX 적용."""
        if self.motor_angle_at is None or time.monotonic() - self.motor_angle_at > p['MOTOR_TIMEOUT']:
            return 0.0
        real_deg = self.motor_angle * p['ANGLE_SCALE']
        eff_deg = float(np.clip(real_deg * p['ROT_GAIN'], -p['ROT_MAX'], p['ROT_MAX']))
        return math.radians(eff_deg)

    # ── 클러스터링 (콘 게이트 판정 전용, 조향 계산과 무관) ──
    def _clusters(self, r, logical_deg, p, heading):
        idx = np.where(np.isfinite(r) & (r > p['MIN_DIST']) & (r < p['MAX_DIST']))[0]
        th = np.radians(logical_deg[idx])
        xs = -r[idx] * np.sin(th)   # cone_avoid 좌표 규칙과 동일 (draw()의 m2px 입력과 일치)
        ys = r[idx] * np.cos(th)
        degs = logical_deg[idx]

        # 진행방향 예측 전단: 근거리는 그대로 두고 멀어질수록만 중심선이 옆으로 밀림
        k = math.tan(heading)
        xs_shear = xs - k * ys

        # 평행사변형 ROI (|x'| <= ROI_X, |y| <= ROI_Y, 전단된 좌표 기준) - 주변 잡음 제거
        roi = (np.abs(xs_shear) <= p['ROI_X']) & (np.abs(ys) <= p['ROI_Y'])
        xs, ys, degs = xs[roi], ys[roi], degs[roi]

        # 클러스터 분리: 연속 포인트간 거리를 벡터로 한 번에 계산해서 구간만 나눔
        # (포인트 수만큼이 아니라 클러스터 수만큼만 파이썬 루프)
        n = len(xs)
        gaps = np.hypot(np.diff(xs), np.diff(ys))
        split_at = np.where(gaps > p['GAP'])[0] + 1
        starts = np.concatenate(([0], split_at))
        ends = np.concatenate((split_at, [n]))

        clusters = []
        for s, e in zip(starts, ends):
            if e - s >= p['CLU_PTS']:
                clusters.append(list(zip(xs[s:e], ys[s:e], degs[s:e])))
        return clusters

    @staticmethod
    def _cluster_width(cluster):
        xs = [pt[0] for pt in cluster]
        return max(xs) - min(xs)

    def _gate_state(self, clusters, p):
        """폭이 콘 크기인 클러스터가 왼쪽/오른쪽 섹터에 각각 몇 개 있는지"""
        left_n = right_n = 0
        for c in clusters:
            w = self._cluster_width(c)
            if w < p['CONE_W_MIN'] or w > p['CONE_W']:
                continue
            mean_deg = sum(pt[2] for pt in c) / len(c)
            if p['L_DEG1'] <= mean_deg <= p['L_DEG2']:
                left_n += 1
            elif p['R_DEG1'] <= mean_deg <= p['R_DEG2']:
                right_n += 1
        return left_n, right_n

    def _update_gate(self, gate_both, gate_either, p):
        """진입: 좌우 동시(GATE_N) 확인 CONFIRM번 연속 -> active=True.
        유지: active 중엔 좌/우 한쪽만 GATE_N 이상이어도 유지 타이머 리셋(EXIT_SEC 재시작 방지).
        좌우 둘 다 GATE_N 미만인 상태가 EXIT_SEC 동안 계속돼야 False로 이탈."""
        if gate_both:
            self.gate_confirm_count += 1
            self.gate_clear_since = None
            if self.gate_confirm_count >= p['CONFIRM']:
                self.cone_active = True
        else:
            self.gate_confirm_count = 0
            if self.cone_active:
                if gate_either:
                    self.gate_clear_since = None
                else:
                    now = time.monotonic()
                    if self.gate_clear_since is None:
                        self.gate_clear_since = now
                    elif now - self.gate_clear_since >= p['EXIT_SEC']:
                        self.cone_active = False

    # ── 콜백 ─────────────────────────────────────────────
    def cb(self, msg):
        p = self._read_params()
        r = np.array(msg.ranges, dtype=np.float32)
        n = len(r)
        logical_deg = self._logical_deg(n, p)

        l_dist = self.sector_dist(r, logical_deg, p['L_DEG1'], p['L_DEG2'], p['MIN_DIST'], p['MAX_DIST'])
        r_dist = self.sector_dist(r, logical_deg, p['R_DEG1'], p['R_DEG2'], p['MIN_DIST'], p['MAX_DIST'])

        # P 제어: 오른쪽 가까우면 왼쪽으로 (-), 왼쪽 가까우면 오른쪽으로 (+)
        raw = float(np.clip(p['KP'] * (r_dist - l_dist), -MAX_ANGLE, MAX_ANGLE))

        # EMA 평활화
        self.prev_angle = p['EMA'] * raw + (1.0 - p['EMA']) * self.prev_angle
        angle = self.prev_angle

        heading = self._path_heading_rad(p)

        clusters = self._clusters(r, logical_deg, p, heading)
        left_n, right_n = self._gate_state(clusters, p)
        gate_both = (left_n >= p['GATE_N']) and (right_n >= p['GATE_N'])
        gate_either = (left_n >= p['GATE_N']) or (right_n >= p['GATE_N'])
        self._update_gate(gate_both, gate_either, p)
        self.pub_cmd(angle)   # /cmd/cone을 active보다 먼저 발행 (mux 레이스 방지)
        self.active_pub.publish(Bool(data=self.cone_active))

        self._viz = (r, logical_deg, l_dist, r_dist, angle, p, clusters, left_n, right_n, heading)

    def _render_timer(self):
        if self._viz is not None:
            self.draw(*self._viz)

    # ── 시각화 ────────────────────────────────────────────
    def draw(self, r, logical_deg, l_dist, r_dist, angle, p, clusters, left_n, right_n, heading):
        img = np.full((VH, VW, 3), 245, np.uint8)

        # 거리 그리드
        for d in range(1, int(VIEW) + 1):
            row = VH - int(d * SCL)
            if 0 <= row < VH:
                cv2.line(img, (0, row), (VW, row), (215, 215, 215), 1)
                cv2.putText(img, f'{d}m', (2, row - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (175, 175, 175), 1)
        cv2.line(img, (VW // 2, 0), (VW // 2, VH), (215, 215, 215), 1)

        # ROI 평행사변형 (동적 전단 반영, |x'| <= ROI_X, |y| <= ROI_Y - 화면엔 역전단해서 그림: x=x'+k*y)
        k = math.tan(heading)
        corners = [(-p['ROI_X'], -p['ROI_Y']), (p['ROI_X'], -p['ROI_Y']),
                   (p['ROI_X'], p['ROI_Y']), (-p['ROI_X'], p['ROI_Y'])]
        pts = [m2px(cx + k * cy, cy) for cx, cy in corners]
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, (0, 130, 0), 1)

        # LiDAR 포인트 (섹터별 색상, 보정된 각도 기준으로 표시)
        for i in range(len(r)):
            ri = r[i]
            if not np.isfinite(ri) or ri < p['MIN_DIST'] or ri > VIEW:
                continue
            deg = float(logical_deg[i])
            th  = np.radians(deg)
            px  = m2px(-ri * np.sin(th), ri * np.cos(th))
            if not (0 <= px[0] < VW and 0 <= px[1] < VH):
                continue
            if p['L_DEG1'] <= deg <= p['L_DEG2']:
                cv2.circle(img, px, 3, (30, 130, 255), -1)   # 파랑=왼쪽
            elif p['R_DEG1'] <= deg <= p['R_DEG2']:
                cv2.circle(img, px, 3, (0, 200, 100), -1)    # 초록=오른쪽
            else:
                cv2.circle(img, px, 2, (195, 195, 195), -1)  # 회색=기타

        # 섹터 대표 거리 화살표
        for dist, mid_d, col in [
            (l_dist, (p['L_DEG1'] + p['L_DEG2']) / 2.0, (30,  130, 255)),
            (r_dist, (p['R_DEG1'] + p['R_DEG2']) / 2.0, ( 0,  200, 100)),
        ]:
            th = mid_d * np.pi / 180.0
            ep = m2px(-dist * np.sin(th), dist * np.cos(th))
            if 0 <= ep[0] < VW and 0 <= ep[1] < VH:
                cv2.arrowedLine(img, CPX, ep, col, 2, tipLength=0.15)
                cv2.circle(img, ep, 8, col, -1)

        # 콘으로 인정된 클러스터 강조(자홍색 테두리)
        for c in clusters:
            w = self._cluster_width(c)
            if w < p['CONE_W_MIN'] or w > p['CONE_W']:
                continue
            cx = sum(pt[0] for pt in c) / len(c)
            cy = sum(pt[1] for pt in c) / len(c)
            cp = m2px(cx, cy)
            if 0 <= cp[0] < VW and 0 <= cp[1] < VH:
                cv2.circle(img, cp, 10, (200, 0, 200), 2)

        # 차량 아이콘 + 조향 화살표
        cv2.circle(img, CPX, 6, (0, 0, 180), -1)
        slen = 35
        sx = int(CPX[0] + slen * np.sin(np.radians(angle)))
        sy = int(CPX[1] - slen * np.cos(np.radians(angle)))
        cv2.arrowedLine(img, CPX, (sx, sy), (0, 0, 220), 3, tipLength=0.35)

        # HUD 텍스트
        cv2.putText(img,
            f'L:{l_dist:.2f}m  R:{r_dist:.2f}m  angle:{angle:.1f}deg  KP:{p["KP"]:.0f}  '
            f'yaw:{p["YAW_OFFSET"]:.0f}  mirror:{int(p["MIRROR"])}',
            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 1)
        gate_col = (0, 150, 0) if self.cone_active else (120, 120, 120)
        cv2.putText(img,
            f'gate L:{left_n}/{p["GATE_N"]} R:{right_n}/{p["GATE_N"]}  active:{int(self.cone_active)}  '
            f'path_heading:{math.degrees(heading):.0f}deg',
            (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, gate_col, 1)
        if self.cone_active:
            cv2.rectangle(img, (VW - 130, 4), (VW - 4, 30), (0, 150, 0), 2)
            cv2.putText(img, 'CONE GATE', (VW - 122, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
        cv2.imshow(VIEW_WIN, img)
        cv2.waitKey(1)

    # ── 모터 퍼블리시 ─────────────────────────────────────
    def pub_cmd(self, angle):
        # active일 때만 /cmd/cone 발행 (mux가 active 신호+이 명령을 같이 봄)
        if not self.cone_active:
            return
        msg = Float32MultiArray()
        msg.data = [float(np.clip(angle, -MAX_ANGLE, MAX_ANGLE)), float(CONE_SPEED)]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ConeAvoidNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Float32MultiArray()
        stop.data = [0.0, 0.0]
        node.pub.publish(stop)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
