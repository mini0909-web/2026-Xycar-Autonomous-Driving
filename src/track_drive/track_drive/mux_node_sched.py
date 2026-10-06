#!/usr/bin/env python3
# mux_node_sched.py - 대회용: E2E 차선주행(부스트 가속) + 라바콘 회피 + 동적 장애물 회피 MUX
#   우선순위: traffic_light(/traffic_light/active + /cmd/traffic_light) > cone > dynamic > lane
#   LANE 속도: 조향이 ANGLE_THRESH 이내로 BOOST_CONFIRM_FRAMES 연속되면(직진 확정)
#              target_speed(Q/E, 기본 MAX_SPEED)까지 부스트, 벗어나면 BASE_SPEED로 복귀.
#   모든 모드(LANE/CONE/DYNAMIC) 가속은 프레임(틱, 50Hz)당 SPEED_STEP_PER_FRAME(=0.2)로 제한.
#   감속/정지는 안전상 즉시(장애물 회피 진입이 늦어지면 위험하므로).
#   S: 주행 정지  R: 주행 재개  Q: 부스트 상한+  E: 부스트 상한-
import time
import rclpy, cv2
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from visualization_msgs.msg import Marker, MarkerArray

# 작은 이미지 연산을 매번 멀티스레드로 돌리면 스레드 생성/동기화 오버헤드가
# 더 커서 오히려 CPU를 더 먹음 (e2e_pure의 torch.set_num_threads(1)과 같은 이유).
cv2.setNumThreads(1)

MAX_SPEED   = 32.0
SPEED_STEP  = 2.0
SPEED_STEP_PER_FRAME = 0.2  # 가속은 매 프레임(틱, 50Hz) 최대 이만큼만 - 감속/정지는 즉시
STARTUP_DELAY_SEC = 5.0  # 노드 시작 후 이 시간 동안은 모터 컨트롤러 워밍업 대기 - 무조건 정지
BASE_SPEED   = 20.0   # 디폴트/바닥 속도 - 조향이 커지면 이 속도까지 되돌아옴
ANGLE_THRESH = 10.0   # -10~10 이내(작은 조향)면 부스트 후보, 벗어나면(큰 조향) 즉시 감속 목표
BOOST_CONFIRM_FRAMES = 10  # -10~10이 이만큼 연속돼야 진짜 직진으로 보고 부스트 시작
                          # (S자 구간에서 각도가 순간적으로 임계값 안쪽으로 스칠 때 오인 방지)
LANE_TIMEOUT_SEC     = 0.5
CONE_TIMEOUT_SEC     = 0.3
ACTIVE_TIMEOUT_SEC   = 0.3
DYNAMIC_TIMEOUT_SEC        = 0.3
DYNAMIC_ACTIVE_TIMEOUT_SEC = 0.3
TL_TIMEOUT_SEC              = 0.3
TL_ACTIVE_TIMEOUT_SEC       = 0.3

WIN = 'MUX Sched Control'
W, H = 700, 300
MARKER_FRAME = 'laser_frame'


def ramp_toward(desired, previous):
    """모든 모드(LANE/CONE/DYNAMIC) 공통 프레임 단위 속도 램프.
    가속만 제한(감속/정지는 즉시 - 장애물 회피 진입이 늦어지면 위험하므로).
    데드존을 건너뛰지 않고 0부터 프레임(틱)당 SPEED_STEP_PER_FRAME만큼만
    순차적으로 증가시킨다 - 모터가 정지 상태에서 중간값을 스킵하고 받으면
    그 스킵 자체가 훅 튀는 느낌을 줘서 힘이 딸리는 것처럼 걸린다."""
    desired = max(0.0, float(desired))
    previous = max(0.0, float(previous))
    if desired <= previous:
        return desired  # 감속/정지는 즉시
    return min(desired, previous + SPEED_STEP_PER_FRAME)

class DriveModeMuxSched(Node):

    def __init__(self):
        super().__init__('drive_mode_mux_sched')
        self.motor_lane = None
        self.lane_received_at = None
        self.cone_command = None
        self.cone_received_at = None
        self.cone_active = False
        self.active_received_at = None
        self.dynamic_command = None
        self.dynamic_received_at = None
        self.dynamic_active = False
        self.dynamic_active_received_at = None
        self.tl_command = None
        self.tl_received_at = None
        self.tl_active = False
        self.tl_active_received_at = None

        self.target_speed   = MAX_SPEED  # Q/E로 조절하는 부스트 상한
        self.cur_speed      = 0.0
        self.estop          = False
        self.last_source     = None
        self.start_time = time.monotonic()
        self.straight_streak = 0  # 조향이 ANGLE_THRESH 이내로 연속된 프레임 수(부스트 시작 조건)
        self._viz = None

        self.pub = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', 10)
        self.create_subscription(Float32MultiArray, '/motor_lane', self.cb_lane, 10)
        self.create_subscription(Float32MultiArray, '/cmd/cone', self.cb_cone, 10)
        self.create_subscription(Bool, '/cone/active', self.cb_cone_active, 10)
        self.create_subscription(Float32MultiArray, '/cmd/dynamic', self.cb_dynamic, 10)
        self.create_subscription(Bool, '/dynamic/active', self.cb_dynamic_active, 10)
        self.create_subscription(Float32MultiArray, '/cmd/traffic_light', self.cb_tl, 10)
        self.create_subscription(Bool, '/traffic_light/active', self.cb_tl_active, 10)
        # MUX 창에 포커스가 있어도 t키로 신호등 출발 게이트를 강제로 열 수 있게
        # 토픽으로 넘긴다 - cv2.waitKey는 프로세스별로 자기 창에 포커스가 있을
        # 때만 키를 잡아서, 이게 없으면 TL-Detector 창을 따로 클릭해야만 했다.
        self.gate_override_pub = self.create_publisher(Bool, '/traffic_light/force_gate', 10)
        self.create_timer(0.02, self.timer_cb)
        # 키입력(cv2.waitKey)/제어/발행은 50Hz 그대로, HUD 그리기만 10Hz로 분리 - CPU 절약
        self.create_timer(0.1, self._render_timer)

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, W, H)
        self.get_logger().info(
            'DriveModeMuxSched ready (E2E boost lane + cone + dynamic) | '
            'S=정지 R=재개 Q/E=부스트상한 T=신호등게이트강제통과'
        )

    def cb_lane(self, msg):
        if len(msg.data) >= 1:
            self.motor_lane = msg
            self.lane_received_at = time.monotonic()

    def cb_cone(self, msg):
        if len(msg.data) >= 2:
            self.cone_command = (float(msg.data[0]), float(msg.data[1]))
            self.cone_received_at = time.monotonic()

    def cb_cone_active(self, msg):
        self.cone_active = bool(msg.data)
        self.active_received_at = time.monotonic()

    def cb_dynamic(self, msg):
        if len(msg.data) >= 2:
            self.dynamic_command = (float(msg.data[0]), float(msg.data[1]))
            self.dynamic_received_at = time.monotonic()

    def cb_dynamic_active(self, msg):
        self.dynamic_active = bool(msg.data)
        self.dynamic_active_received_at = time.monotonic()

    def cb_tl(self, msg):
        if len(msg.data) >= 2:
            self.tl_command = (float(msg.data[0]), float(msg.data[1]))
            self.tl_received_at = time.monotonic()

    def cb_tl_active(self, msg):
        self.tl_active = bool(msg.data)
        self.tl_active_received_at = time.monotonic()

    @staticmethod
    def _fresh(t, timeout):
        return t is not None and time.monotonic() - t <= timeout

    def _publish(self, angle, speed):
        m = Float32MultiArray()
        m.data = [float(angle), float(speed)]
        self.pub.publish(m)

    def _select(self):
        if self.estop:
            self.straight_streak = 0
            return 'ESTOP', 0.0, 0.0
        if self._fresh(self.tl_active_received_at, TL_ACTIVE_TIMEOUT_SEC) and self.tl_active:
            if self._fresh(self.tl_received_at, TL_TIMEOUT_SEC):
                self.straight_streak = 0
                return 'TRAFFIC_LIGHT', self.tl_command[0], self.tl_command[1]
            # active=True 직후 또는 stale command면 아래 source로 폴백한다.
        if self._fresh(self.active_received_at, ACTIVE_TIMEOUT_SEC) and self.cone_active:
            if self._fresh(self.cone_received_at, CONE_TIMEOUT_SEC):
                self.straight_streak = 0
                return 'CONE', self.cone_command[0], self.cone_command[1]
            # active=True 직후 또는 stale command면 아래 source로 폴백한다.
        if self._fresh(self.dynamic_active_received_at, DYNAMIC_ACTIVE_TIMEOUT_SEC) and self.dynamic_active:
            if self._fresh(self.dynamic_received_at, DYNAMIC_TIMEOUT_SEC):
                self.straight_streak = 0
                return 'DYNAMIC', self.dynamic_command[0], self.dynamic_command[1]
            # A stale dynamic command is released to E2E rather than held.
        if self._fresh(self.lane_received_at, LANE_TIMEOUT_SEC) and self.motor_lane is not None:
            angle = float(self.motor_lane.data[0])
            # 조향(즉시값)이 -10~10 이내로 BOOST_CONFIRM_FRAMES 연속돼야 "진짜 직진"으로
            # 보고 부스트 상한(target_speed)을 목표로 잡는다. S자 구간처럼 각도가 임계값
            # 안쪽을 순간적으로 스치기만 하면 연속 카운트가 안 쌓여 부스트가 시작되지
            # 않는다. 큰 조향으로 벗어나면 카운트 리셋 + BASE_SPEED를 목표로 되돌아간다.
            if abs(angle) <= ANGLE_THRESH:
                self.straight_streak += 1
                if self.straight_streak >= BOOST_CONFIRM_FRAMES:
                    desired = self.target_speed
                else:
                    desired = self.cur_speed  # 아직 직진 확정 전 - 현재 속도 유지
            else:
                self.straight_streak = 0
                desired = BASE_SPEED
            return 'LANE', angle, desired
        return 'NO_CMD_STOP', 0.0, 0.0

    def timer_cb(self):
        if time.monotonic() - self.start_time < STARTUP_DELAY_SEC:
            # 모터 컨트롤러 워밍업 대기 - 이 구간엔 어떤 소스가 와도 무조건 정지
            self.cur_speed = 0.0
            self.straight_streak = 0
            self._publish(0.0, 0.0)
            cv2.waitKey(1)
            return
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            if not self.estop:
                self.get_logger().warn('ESTOP: 주행 정지')
            self.estop = True
            self.cur_speed = 0.0
        elif k == ord('r'):
            if self.estop:
                self.get_logger().info('주행 재개 (부스트상한={:.1f})'.format(self.target_speed))
                self.estop = False
                self.cur_speed = 0.0
                # ramp_toward가 0부터 프레임 단위로 순차 가속하므로
                # 여기선 그냥 0에서 시작하기만 하면 된다.
        elif k == ord('q'):
            self.target_speed = min(self.target_speed + SPEED_STEP, MAX_SPEED)
            self.get_logger().info('boost+ target={:.1f}'.format(self.target_speed))
        elif k == ord('e'):
            self.target_speed = max(self.target_speed - SPEED_STEP, BASE_SPEED)
            self.get_logger().info('boost- target={:.1f}'.format(self.target_speed))
        elif k == ord('t'):
            self.get_logger().info('신호등 출발 게이트 강제 통과 요청 전송')
            self.gate_override_pub.publish(Bool(data=True))

        source, angle, req_speed = self._select()

        # LANE/CONE/DYNAMIC 전부 동일한 프레임 단위 램프(ramp_toward)를 거친다.
        # 가속만 제한되고(데드존 스킵 없이 0부터 순차 증가) 감속/정지는 즉시이므로,
        # cone/dynamic 진입 시 안전하게 즉시 감속되고 부스트나 재개처럼 가속이
        # 필요한 전환만 프레임당 1씩 부드럽게 올라간다.
        if source in ('LANE', 'CONE', 'DYNAMIC', 'TRAFFIC_LIGHT'):
            self.cur_speed = ramp_toward(req_speed, self.cur_speed)
            self._publish(angle, self.cur_speed)
        else:
            self.cur_speed = 0.0
            self._publish(0.0, 0.0)

        if source != self.last_source:
            self.get_logger().info('Control source: {}'.format(source))
            self.last_source = source

        self._viz = (source, angle)
        self._publish_markers(source, angle)

    def _publish_markers(self, source, angle):
        """Publish the currently selected control source for RViz2."""
        markers = MarkerArray()
        markers.markers.append(Marker(action=Marker.DELETEALL))

        marker = Marker()
        marker.header.frame_id = MARKER_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'drive_mux_sched'
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.5
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.8
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.25
        marker.color.a = 1.0
        colors = {
            'ESTOP': (1.0, 0.0, 0.0),
            'TRAFFIC_LIGHT': (0.6, 0.0, 1.0),
            'CONE': (1.0, 0.5, 0.0),
            'DYNAMIC': (1.0, 1.0, 0.0),
            'LANE': (0.0, 1.0, 0.0),
            'NO_CMD_STOP': (1.0, 0.0, 0.0),
        }
        marker.color.r, marker.color.g, marker.color.b = colors.get(source, (1.0, 1.0, 1.0))
        marker.text = 'MUX_SCHED: {}\nangle: {:+.1f}  speed: {:.1f}'.format(
            source, float(angle), self.cur_speed)
        markers.markers.append(marker)
        self.marker_pub.publish(markers)

    def _render_timer(self):
        if self._viz is not None:
            self._draw_hud(*self._viz)

    def _draw_hud(self, source, angle):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        colors = {
            'ESTOP': (0, 0, 255),
            'TRAFFIC_LIGHT': (200, 0, 160),
            'CONE': (0, 80, 220),
            'DYNAMIC': (0, 150, 220),
            'LANE': (0, 150, 0), 'NO_CMD_STOP': (0, 90, 200),
        }
        col = colors.get(source, (90, 90, 90))
        cv2.rectangle(img, (0, 0), (W, 70), col, 3)
        cv2.putText(img, source, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(img, 'angle:{:+6.1f}  cur:{:.1f} -> boost_cap:{:.1f}'.format(
                    angle, self.cur_speed, self.target_speed),
                    (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 0), 2)
        cv2.putText(img, 'straight_streak:{}/{}  base:{:.0f}  thresh:+-{:.0f}'.format(
                    self.straight_streak, BOOST_CONFIRM_FRAMES, BASE_SPEED, ANGLE_THRESH),
                    (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
        cv2.putText(img, 'cone_active:{}  dynamic_active:{}'.format(
                    int(self.cone_active), int(self.dynamic_active)),
                    (15, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
        cv2.putText(img, 'S:stop  R:resume  Q:boost+  E:boost-  T:tl-gate',
                    (15, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        cv2.imshow(WIN, img)


def main(args=None):
    rclpy.init(args=args)
    node = DriveModeMuxSched()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish(0.0, 0.0)
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
