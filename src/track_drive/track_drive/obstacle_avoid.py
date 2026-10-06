#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obstacle_avoid.py - 라이다 전용 정적 장애물 회피 (카메라 미사용, 시간기반 상태머신)
  - 정면 좁은 폭 안에 클러스터가 CONFIRM프레임 연속 잡히면 발동
  - 방향은 장애물 클러스터 자신의 좌우 위치(x 부호)로 결정 (반대쪽으로 조향)
  - CHANGE_OUT -> RECENTER -> LANE_FOLLOW 전부 고정 시간으로 전환 (카메라 확인 없음)
  - 좌표계/YAW_OFF 관례는 cone_avoid.py와 동일 (x=좌우, y=전방, x=-r*sin(th), y=r*cos(th))
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

MAX_ANGLE = 100

DEFAULTS = {
    'YAW_OFF':   0,   # cone_avoid와 같은 라이다 보정값 (실제값 = 트랙바값-180)
    'MIRROR':    0,
    'MIN_CM':   15,   # 유효 최소 거리(cm)
    'MAX_CM':  600,   # 유효 최대 거리(cm)
    # ── 클러스터링 (cone_avoid와 동일 파라미터명이지만 값은 독립적) ──
    # 큰 물체는 표면이 평평하지 않거나 각도가 틀어지면 라이다 포인트가 중간중간
    # 듬성듬성 찍혀서 여러 조각으로 쪼개져 보일 수 있음 - 콘처럼 정밀 구분할 필요
    # 없는 obstacle_avoid는 넉넉하게 잡아서 조각난 걸 하나로 합침.
    'GAP_CM':   35,   # 클러스터 분리 간격(cm)
    'CLU_PTS':   3,   # 클러스터 최소 포인트 수
    # ── 전방 트리거 게이트 (이 좁은 폭 안에 뭔가 있으면 장애물로 봄) ──
    'FRONT_X_CM': 30,   # 정면 게이트 좌우 반폭(cm)
    'FRONT_Y_CM': 100,  # 정면 게이트 전방 거리(cm)
    'FRONT_MIN_PTS': 3, # 게이트 안 클러스터 최소 포인트 수
    'OBSTACLE_MIN_CM': 18, # 이 폭(cm) 미만은 콘으로 보고 무시 (실측 디버그로 확인 후 조정)
    'CONFIRM':   3,     # 발동에 필요한 연속 프레임 수
    'EXIT_MS':   5,     # 게이트 이탈 유지시간(x100ms), 5=0.5초 (노이즈로 순간 끊길 때 debounce)
    # ── 안전 임계값 ───────────────────────────────────────
    'STOP_CM':  15,   # 발동 시점에 장애물이 이 거리(cm) 이내면 스윙 대신 즉시 정지
    'CLEAR_CM': 10,   # 스윙하려는 방향에 이 거리(cm) 이내 장애물 있으면 정지 (스윙 취소)
    # ── 조향/속도 ─────────────────────────────────────────
    'BIAS_DEG': 40,   # 회피 조향값(고정, 반대쪽으로)
    'SPEED':     7,   # 회피 중 속도
    'MAX_STEER': 50,  # 조향 클립 한계
    # ── 상태 지속시간 (x100ms) ────────────────────────────
    'CHANGE_MS': 20,  # CHANGE_OUT 유지시간, 20=2.0초
    'RECENTER_MS': 4, # RECENTER 유지시간, 4=0.4초
    'COOLDOWN_MS': 10, # 복귀 후 재발동 억제시간, 10=1.0초
    # ── 동적 ROI (커브 중 진행방향 예측 전단(shear), cone_avoid와 동일 방식) ──
    'ANGLE_SCALE_PCT': 20, # raw angle(-100~100) -> 실제 조향각(도) 환산 비율. 실측: raw 100 ~= 실제 20도
    'ROT_GAIN_PCT': 100,  # 환산된 실제각도 대비 게이트 기울임 비율(%)
    'ROT_MAX_DEG':  30,   # 게이트 기울임각 상한(도, 환산 후 기준)
    # /motor_lane(e2e) - obstacle_avoid 자신의 CHANGE_OUT 조향(base+bias)에만 씀
    'LANE_TIMEOUT_MS': 5, # /motor_lane 최신값 유효시간(x100ms), 끊기면 base 0
    'LANE_EMA_PCT': 30,   # e2e 각도 스무딩(%)
    # /xycar_motor(mux 최종출력) - ROI 전단(heading)에만 씀. 지금 실제로 차가
    # 받고 있는 명령 그대로라 LANE/CONE/OBSTACLE/ESTOP 상태를 따로 안 구분해도 됨.
    'MOTOR_TIMEOUT_MS': 5, # /xycar_motor 최신값 유효시간(x100ms), 끊기면 heading 0
    'MOTOR_EMA_PCT': 30,   # 스무딩(%). 낮을수록 순간 오실레이션을 더 눌러줌(진짜 커브는 여전히 따라감)
}
TRACKBAR_MAX = {
    'YAW_OFF': 360, 'MIRROR': 1,
    'MIN_CM': 200, 'MAX_CM': 1000,
    'GAP_CM': 100, 'CLU_PTS': 20,
    'FRONT_X_CM': 200, 'FRONT_Y_CM': 400, 'FRONT_MIN_PTS': 20, 'OBSTACLE_MIN_CM': 100,
    'CONFIRM': 20, 'EXIT_MS': 50,
    'STOP_CM': 100, 'CLEAR_CM': 100,
    'BIAS_DEG': 90, 'SPEED': 30, 'MAX_STEER': 100,
    'CHANGE_MS': 100, 'RECENTER_MS': 50, 'COOLDOWN_MS': 50,
    'ANGLE_SCALE_PCT': 100, 'ROT_GAIN_PCT': 200, 'ROT_MAX_DEG': 90, 'LANE_TIMEOUT_MS': 50,
    'LANE_EMA_PCT': 100, 'MOTOR_TIMEOUT_MS': 50, 'MOTOR_EMA_PCT': 100,
}

# 트랙바 창 자체를 안 띄우고 DEFAULTS 고정값을 씀 (CPU/창 오버헤드 절약).
# 국민대 등에서 튜닝할 때만 이 노드에 한해 True로 바꿔서 씀.
ENABLE_TRACKBAR = False

VIEW = 1.5
SCL  = 200
VH   = int(VIEW * SCL)
VW   = VH * 2
CPX  = (VW // 2, VH - 5)
PARAMS_WIN = 'Obstacle Params'
VIEW_WIN   = 'Obstacle Avoid'


def m2px(x, y):
    return int(VW // 2 + x * SCL), int(VH - y * SCL)


def _nop(_):
    pass


class ObstacleAvoidNode(Node):

    def __init__(self):
        super().__init__('obstacle_avoid')
        # mux가 /xycar_motor의 유일한 발행자. 이 노드는 /cmd/obstacle + /obstacle/active만 내보냄
        self.pub = self.create_publisher(Float32MultiArray, '/cmd/obstacle', 10)
        self.active_pub = self.create_publisher(Bool, '/obstacle/active', 10)
        self.create_subscription(
            LaserScan, '/scan', self.cb, qos_profile_sensor_data)
        # CHANGE_OUT 조향(base+bias)용: e2e 차선주행 각도
        self.create_subscription(
            Float32MultiArray, '/motor_lane', self.cb_lane, 10)
        self.lane_angle = 0.0
        self.lane_angle_at = None
        # 동적 게이트 회전용: mux가 지금 실제로 내보내는 최종 명령 각도
        self.create_subscription(
            Float32MultiArray, '/xycar_motor', self.cb_motor, 10)
        self.motor_angle = 0.0
        self.motor_angle_at = None
        # cone이 활성 상태면 트리거 조건이 안 겹치게 완전히 양보 (mux 우선순위와 별개의 방어)
        self.create_subscription(Bool, '/cone/active', self.cb_cone_active, 10)
        self.cone_active = False

        # 정면 게이트 hysteresis 상태
        self.gate_confirm_count = 0
        self.gate_clear_since = None
        self.gate_active = False

        # 회피 상태머신
        self.state = 'LANE_FOLLOW'
        self.state_started_at = time.monotonic()
        self.direction = 0.0   # +1=오른쪽으로 스윙, -1=왼쪽으로 스윙
        self.cooldown_until = None

        if ENABLE_TRACKBAR:
            cv2.namedWindow(PARAMS_WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PARAMS_WIN, 360, 480)
            for name, val in DEFAULTS.items():
                cv2.createTrackbar(name, PARAMS_WIN, val, TRACKBAR_MAX[name], _nop)

        # 시각화는 스캔 콜백(제어)과 분리된 느린 타이머(10Hz)에서만 그림 - CPU 절약
        self._viz = None
        self.create_timer(0.1, self._render_timer)

        self.get_logger().info('준비됨. 라이다 전용 시간기반 회피 (Params 창 트랙바로 튜닝)')

    @staticmethod
    def _g(name):
        return (cv2.getTrackbarPos(name, PARAMS_WIN) if ENABLE_TRACKBAR else DEFAULTS[name])

    def cb_lane(self, msg):
        """e2e /motor_lane 각도를 EMA로 스무딩 - CHANGE_OUT 조향(base+bias)에만 씀."""
        if len(msg.data) >= 1:
            raw = float(msg.data[0])
            alpha = self._g('LANE_EMA_PCT') / 100.0
            if self.lane_angle_at is None:
                self.lane_angle = raw
            else:
                self.lane_angle = alpha * raw + (1.0 - alpha) * self.lane_angle
            self.lane_angle_at = time.monotonic()

    def cb_motor(self, msg):
        """mux 최종출력 /xycar_motor 각도를 EMA로 스무딩 - ROI 전단(heading)에만 씀.
        지금 실제로 차가 받는 명령 그대로라 LANE/CONE/OBSTACLE/ESTOP 구분 없이 그냥 이거 하나로 충분."""
        if len(msg.data) >= 1:
            raw = float(msg.data[0])
            alpha = self._g('MOTOR_EMA_PCT') / 100.0
            if self.motor_angle_at is None:
                self.motor_angle = raw
            else:
                self.motor_angle = alpha * raw + (1.0 - alpha) * self.motor_angle
            self.motor_angle_at = time.monotonic()

    def cb_cone_active(self, msg):
        self.cone_active = bool(msg.data)

    def _read_params(self):
        g = (lambda n: cv2.getTrackbarPos(n, PARAMS_WIN)) if ENABLE_TRACKBAR else (lambda n: DEFAULTS[n])
        return {
            'YAW_OFFSET': float(g('YAW_OFF') - 180),
            'MIRROR': bool(g('MIRROR')),
            'MIN_DIST': g('MIN_CM') / 100.0,
            'MAX_DIST': g('MAX_CM') / 100.0,
            'GAP': g('GAP_CM') / 100.0,
            'CLU_PTS': max(1, g('CLU_PTS')),
            'FRONT_X': g('FRONT_X_CM') / 100.0,
            'FRONT_Y': g('FRONT_Y_CM') / 100.0,
            'FRONT_MIN_PTS': max(1, g('FRONT_MIN_PTS')),
            'OBSTACLE_MIN_WIDTH': g('OBSTACLE_MIN_CM') / 100.0,
            'CONFIRM': max(1, g('CONFIRM')),
            'EXIT_SEC': g('EXIT_MS') / 10.0,
            'STOP_DIST': g('STOP_CM') / 100.0,
            'CLEAR_DIST': g('CLEAR_CM') / 100.0,
            'BIAS': float(g('BIAS_DEG')),
            'SPEED': float(g('SPEED')),
            'MAX_STEER': float(g('MAX_STEER')),
            'CHANGE_SEC': g('CHANGE_MS') / 10.0,
            'RECENTER_SEC': g('RECENTER_MS') / 10.0,
            'COOLDOWN_SEC': g('COOLDOWN_MS') / 10.0,
            'ANGLE_SCALE': g('ANGLE_SCALE_PCT') / 100.0,
            'ROT_GAIN': g('ROT_GAIN_PCT') / 100.0,
            'ROT_MAX': float(g('ROT_MAX_DEG')),
            'LANE_TIMEOUT': g('LANE_TIMEOUT_MS') / 10.0,
            'MOTOR_TIMEOUT': g('MOTOR_TIMEOUT_MS') / 10.0,
        }

    # ── 라이다 방향 보정 (cone_avoid.py와 동일) ────────────
    def _logical_deg(self, n, p):
        raw_deg = np.arange(n, dtype=np.float32) * 360.0 / n
        if p['MIRROR']:
            raw_deg = (-raw_deg) % 360.0
        return (raw_deg + p['YAW_OFFSET']) % 360.0

    # ── 진행방향 예측 각도 (mux 최종출력 /xycar_motor 기반, cone_avoid와 동일) ──
    def _path_heading_rad(self, p):
        """mux가 지금 실제로 내보내는 명령 각도(raw 단위)를 그대로 씀 - LANE이든
        CONE이든 OBSTACLE(자기 자신)이든 ESTOP이든 구분할 필요 없이 이거 하나로 충분.
        raw 단위(-100~100)는 ANGLE_SCALE로 실제 조향각(도)으로 환산 후 ROT_GAIN/ROT_MAX 적용."""
        if self.motor_angle_at is None or time.monotonic() - self.motor_angle_at > p['MOTOR_TIMEOUT']:
            return 0.0
        real_deg = self.motor_angle * p['ANGLE_SCALE']
        eff_deg = float(np.clip(real_deg * p['ROT_GAIN'], -p['ROT_MAX'], p['ROT_MAX']))
        return math.radians(eff_deg)

    def _fresh_lane_angle(self, p):
        """e2e의 /motor_lane 각도(raw 단위). 끊긴 지 오래면 0."""
        if self.lane_angle_at is None or time.monotonic() - self.lane_angle_at > p['LANE_TIMEOUT']:
            return 0.0
        return self.lane_angle

    def _change_out_angle(self, p):
        """CHANGE_OUT 중 실제 발행 각도: e2e 차선 조향값에 회피 바이어스를 더함(고정값 아님)."""
        return float(np.clip(self._fresh_lane_angle(p) + self.direction * p['BIAS'],
                              -p['MAX_STEER'], p['MAX_STEER']))

    # ── 클러스터링 (cone_avoid._clusters와 동일 로직, ROI 없이 전체) ──
    def _clusters(self, r, logical_deg, p):
        idx = np.where(np.isfinite(r) & (r > p['MIN_DIST']) & (r < p['MAX_DIST']))[0]
        th = np.radians(logical_deg[idx])
        xs = -r[idx] * np.sin(th)
        ys = r[idx] * np.cos(th)

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
                clusters.append(list(zip(xs[s:e], ys[s:e])))
        return clusters

    @staticmethod
    def _center(cluster):
        xs = [pt[0] for pt in cluster]
        ys = [pt[1] for pt in cluster]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    @staticmethod
    def _min_dist(cluster):
        return min(math.hypot(x, y) for x, y in cluster)

    @staticmethod
    def _cluster_width(cluster):
        xs = [pt[0] for pt in cluster]
        return max(xs) - min(xs)

    @staticmethod
    def _shear(x, y, heading):
        """진행방향 예측각(heading, rad)만큼 평행사변형으로 전단(shear) 변환.
        근거리(y~0)는 그대로 두고, 멀어질수록(y 커질수록)만 중심선이 옆으로 밀림.
        heading>0(우회전 조향)이면 먼 쪽일수록 게이트가 오른쪽으로 밀린다."""
        k = math.tan(heading)
        return x - k * y, y

    # ── 정면 게이트: 좁은 폭 안의 가장 가까운 클러스터 (진행방향 전단 반영, 콘 크기는 제외) ──
    def _front_target(self, clusters, p, heading):
        candidates = []
        for c in clusters:
            if len(c) < p['FRONT_MIN_PTS']:
                continue
            if self._cluster_width(c) < p['OBSTACLE_MIN_WIDTH']:
                continue   # 콘만한 폭은 obstacle 트리거로 안 침 (cone_avoid가 처리)
            cx, cy = self._center(c)
            rx, ry = self._shear(cx, cy, heading)
            if abs(rx) <= p['FRONT_X'] and 0.0 <= ry <= p['FRONT_Y']:
                candidates.append(c)
        if not candidates:
            return None
        return min(candidates, key=self._min_dist)

    def _update_gate(self, gate_now, p):
        """진입은 CONFIRM연속, 이탈은 EXIT_SEC 유지 후 (cone_avoid와 동일 패턴)"""
        if gate_now:
            self.gate_confirm_count += 1
            self.gate_clear_since = None
            if self.gate_confirm_count >= p['CONFIRM']:
                self.gate_active = True
        else:
            self.gate_confirm_count = 0
            if self.gate_active:
                now = time.monotonic()
                if self.gate_clear_since is None:
                    self.gate_clear_since = now
                elif now - self.gate_clear_since >= p['EXIT_SEC']:
                    self.gate_active = False

    # ── 스윙 방향 쪽에 이미 장애물이 있는지 (라이다 거리로만 판단, 진행방향 전단 반영) ──
    def _target_side_blocked(self, clusters, direction, p, heading):
        target_sign = 1.0 if direction > 0 else -1.0
        for c in clusters:
            cx, cy = self._center(c)
            rx, ry = self._shear(cx, cy, heading)
            if ry < 0.0:
                continue
            if (1.0 if rx > 0 else -1.0) == target_sign and self._min_dist(c) <= p['CLEAR_DIST']:
                return True
        return False

    def _elapsed(self):
        return time.monotonic() - self.state_started_at

    def _transition(self, state):
        if state == self.state:
            return
        self.get_logger().info('%s -> %s' % (self.state, state))
        self.state = state
        self.state_started_at = time.monotonic()

    # ── 콜백 ─────────────────────────────────────────────
    def cb(self, msg):
        p = self._read_params()
        r = np.array(msg.ranges, dtype=np.float32)
        n = len(r)
        logical_deg = self._logical_deg(n, p)

        heading = self._path_heading_rad(p)
        clusters = self._clusters(r, logical_deg, p)
        target = self._front_target(clusters, p, heading)
        # cone이 활성 상태면 게이트 자체를 안 채움 - 트리거 조건이 cone과 절대 안 겹치게
        gate_now = target is not None and not self.cone_active
        self._update_gate(gate_now, p)

        now = time.monotonic()
        in_cooldown = self.cooldown_until is not None and now < self.cooldown_until

        if self.cone_active:
            # cone에 완전히 양보: 상태/쿨다운 리셋하고 비활성으로 대기 (mux 우선순위와 별개의 방어)
            self.state = 'LANE_FOLLOW'
            self.gate_active = False
            self.cooldown_until = None
            angle, speed, active = 0.0, 0.0, False
        elif self.state == 'LANE_FOLLOW':
            # gate_active는 EXIT_MS 유예 중엔 target=None이어도 True로 남아있을 수 있음 -
            # 이번 스캔에 실제 target이 있을 때만 발동 판정 (없으면 다음 스캔까지 대기)
            if self.gate_active and not in_cooldown and target is not None:
                target_dist = self._min_dist(target)
                cx, cy = self._center(target)
                rx, _ = self._shear(cx, cy, heading)
                self.direction = -1.0 if rx > 0 else 1.0   # 장애물 반대쪽으로 (전단된 프레임 기준)
                if target_dist <= p['STOP_DIST'] or self._target_side_blocked(clusters, self.direction, p, heading):
                    self._transition('EMERGENCY_STOP')
                else:
                    self._transition('CHANGE_OUT')
            angle, speed, active = 0.0, 0.0, False

        elif self.state == 'CHANGE_OUT':
            if self._target_side_blocked(clusters, self.direction, p, heading):
                self._transition('EMERGENCY_STOP')
                angle, speed, active = 0.0, 0.0, True
            elif self._elapsed() >= p['CHANGE_SEC']:
                self._transition('RECENTER')
                angle, speed, active = 0.0, p['SPEED'], True
            else:
                angle = self._change_out_angle(p)
                speed, active = p['SPEED'], True

        elif self.state == 'RECENTER':
            if self._elapsed() >= p['RECENTER_SEC']:
                self._transition('LANE_FOLLOW')
                self.gate_active = False
                self.gate_confirm_count = 0
                self.cooldown_until = now + p['COOLDOWN_SEC']
                angle, speed, active = 0.0, 0.0, False
            else:
                angle, speed, active = 0.0, p['SPEED'], True

        else:  # EMERGENCY_STOP - 게이트 해제될 때까지 정지 유지
            if not self.gate_active:
                self._transition('LANE_FOLLOW')
                self.cooldown_until = now + p['COOLDOWN_SEC']
                angle, speed, active = 0.0, 0.0, False
            else:
                angle, speed, active = 0.0, 0.0, True

        self.active_pub.publish(Bool(data=active))
        if active:
            m = Float32MultiArray()
            m.data = [float(angle), float(speed)]
            self.pub.publish(m)

        self._viz = (r, logical_deg, p, clusters, target, angle, active, heading)

    def _render_timer(self):
        if self._viz is not None:
            self.draw(*self._viz)

    # ── 시각화 ────────────────────────────────────────────
    def draw(self, r, logical_deg, p, clusters, target, angle, active, heading):
        img = np.full((VH, VW, 3), 245, np.uint8)

        for d in range(1, int(VIEW) + 1):
            row = VH - int(d * SCL)
            if 0 <= row < VH:
                cv2.line(img, (0, row), (VW, row), (215, 215, 215), 1)
                cv2.putText(img, f'{d}m', (2, row - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (175, 175, 175), 1)
        cv2.line(img, (VW // 2, 0), (VW // 2, VH), (215, 215, 215), 1)

        # 좌우 판정 기준선 (전단된 중심선, x = k*y) - 게이트/방향판정이 실제로 보는 축
        k = math.tan(heading)
        heading_end = m2px(k * VIEW, VIEW)
        cv2.line(img, CPX, heading_end, (0, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(img, 'L/R ref', (heading_end[0] + 4, heading_end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 160, 160), 1)

        # 정면 게이트 평행사변형 (동적 전단 반영 - 화면엔 역전단해서 그림: x = x'+k*y)
        corners = [(-p['FRONT_X'], 0.0), (p['FRONT_X'], 0.0),
                   (p['FRONT_X'], p['FRONT_Y']), (-p['FRONT_X'], p['FRONT_Y'])]
        gate_pts = [m2px(gx + k * gy, gy) for gx, gy in corners]
        cv2.polylines(img, [np.array(gate_pts, dtype=np.int32)], True, (0, 130, 0), 1)

        for i in range(len(r)):
            ri = r[i]
            if not np.isfinite(ri) or ri < p['MIN_DIST'] or ri > VIEW:
                continue
            th = np.radians(float(logical_deg[i]))
            px = m2px(-ri * np.sin(th), ri * np.cos(th))
            if 0 <= px[0] < VW and 0 <= px[1] < VH:
                cv2.circle(img, px, 2, (195, 195, 195), -1)

        for c in clusters:
            cx, cy = self._center(c)
            cp = m2px(cx, cy)
            width_cm = self._cluster_width(c) * 100.0
            passes_width = width_cm >= p['OBSTACLE_MIN_WIDTH'] * 100.0
            if c is target:
                col = (0, 0, 220)      # 빨강 = 지금 발동 타겟
            elif passes_width:
                col = (0, 130, 255)    # 주황 = 폭 조건 통과(게이트 밖이라 타겟은 아님)
            else:
                col = (170, 170, 170)  # 회색 = 폭 미달(콘 크기로 판단, 무시됨)
            if 0 <= cp[0] < VW and 0 <= cp[1] < VH:
                cv2.circle(img, cp, 10, col, 2)
                cv2.putText(img, f'{width_cm:.0f}cm', (cp[0] + 12, cp[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        cv2.circle(img, CPX, 6, (0, 0, 180), -1)
        slen = 35
        sx = int(CPX[0] + slen * np.sin(np.radians(angle)))
        sy = int(CPX[1] - slen * np.cos(np.radians(angle)))
        cv2.arrowedLine(img, CPX, (sx, sy), (0, 0, 220), 3, tipLength=0.35)

        cv2.putText(img, f'state:{self.state}  angle:{angle:.1f}  active:{int(active)}',
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 1)
        gate_col = (0, 150, 0) if self.gate_active else (120, 120, 120)
        cv2.putText(img,
            f'gate_active:{int(self.gate_active)}  confirm:{self.gate_confirm_count}  '
            f'path_heading:{math.degrees(heading):.0f}deg',
            (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, gate_col, 1)
        cv2.putText(img,
            f'OBSTACLE_MIN_CM:{p["OBSTACLE_MIN_WIDTH"]*100:.0f}  (red=target orange=pass gray=cone-size ignored)',
            (5, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)
        if active:
            cv2.rectangle(img, (VW - 150, 4), (VW - 4, 30), (0, 100, 220), 2)
            cv2.putText(img, self.state, (VW - 142, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 220), 1)
        cv2.imshow(VIEW_WIN, img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidNode()
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
