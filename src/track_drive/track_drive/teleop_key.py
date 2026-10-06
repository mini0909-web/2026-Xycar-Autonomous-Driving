#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# teleop_key.py - 키보드 수동조종 노드 (E2E 모방학습용 '정답 주행' 생성)
#   pygame 창에 포커스를 둔 채 게임처럼 운전하면 그 조향·속도가 모터 토픽으로 나간다.
#   조작: W 가속 / S 브레이크 / Q 속도4 / E 속도10 / A 좌풀락 / D 우풀락 / SPACE 급정지 / ESC 종료
#   /xycar/xycar_motor 와 /xycar_motor 두 토픽에 동시 publish(시뮬에 무조건 도달).
#   data_logger 가 같은 토픽을 구독하므로 조향/속도가 그대로 학습 라벨로 기록된다.
#   * 수집 중에는 자율주행을 끄고 teleop_key + data_logger 만 실행한다. *
# ============================================================================

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

# 주행 동역학 파라미터(가속/감속/조향 변화율 등) + 표시 창 크기/색
#   MAX_SPEED: 최고속 / ACCEL,BRAKE,COAST: 가속·감속·자연감속(초당)
#   MAX_ANGLE: 조향 한계 / STEER_RATE,CENTER_RATE: 조향·중앙복귀 속도 / FPS: 루프 주기
MAX_SPEED   = 30.0
MIN_SPEED   = 0.0
ACCEL       = 30.0
BRAKE       = 50.0
COAST       = 15.0
MAX_ANGLE   = 100.0
STEER_RATE  = 260.0
CENTER_RATE = 320.0
FPS         = 50

W, H = 460, 260
WHITE = (235, 235, 235)
GRAY  = (90, 90, 90)
DARK  = (28, 28, 32)
GREEN = (60, 220, 90)
RED   = (230, 70, 70)
BLUE  = (80, 160, 240)


# 키보드 조종 노드: 현재 조향·속도를 모터 토픽 두 곳에 발행한다.
class TeleopKey(Node):

    def __init__(self):
        # 노드 초기화: 모터 발행자(두 토픽) 준비, 조향·속도 초기화
        super().__init__('teleop_key')
        self.pub_motor = self.create_publisher(Float32MultiArray, '/xycar_motor', 10)
        self.angle = 0.0
        self.speed = 0.0
        self.get_logger().info('teleop_key (GTA style) ready - pygame 창에 포커스 두고 WASD')

    def publish(self):
        # 현재 조향·속도를 모터 메시지로 만들어 두 토픽에 동시 발행
        m = Float32MultiArray()
        m.data = [float(self.angle), float(self.speed)]
        self.pub_motor.publish(m)


def draw(screen, font, big, angle, speed, focused=True):
    # 상태 표시: 포커스 여부, 속도 바, 조향 바, 조작 힌트를 pygame 창에 그린다.
    #   (포커스가 NO 면 키 입력이 안 들어오므로 창을 클릭/마우스를 올려둬야 함)
    screen.fill(DARK)
    fcol = GREEN if focused else RED
    ftxt = 'FOCUS: YES (입력 OK)' if focused else 'FOCUS: NO -> 창 클릭/마우스 올려두기'
    screen.blit(font.render(ftxt, True, fcol), (200, 18))

    # 속도 바(가로)
    sx, sy, sw = 30, 60, W - 60
    pygame.draw.rect(screen, GRAY, (sx, sy, sw, 22), 1)
    frac = (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
    frac = max(0.0, min(1.0, frac))
    col = GREEN if speed >= 0 else RED
    pygame.draw.rect(screen, col, (sx, sy, int(sw * frac), 22))
    zero = sx + int(sw * ((0 - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)))
    pygame.draw.line(screen, WHITE, (zero, sy - 3), (zero, sy + 25), 1)
    screen.blit(font.render('SPEED {:+5.1f}'.format(speed), True, WHITE), (sx, sy - 26))

    # 조향 바(가로, 중앙 0 기준 좌/우)
    ax, ay, aw = 30, 140, W - 60
    pygame.draw.rect(screen, GRAY, (ax, ay, aw, 22), 1)
    cx = ax + aw // 2
    pygame.draw.line(screen, WHITE, (cx, ay - 3), (cx, ay + 25), 1)
    af = max(-1.0, min(1.0, angle / MAX_ANGLE))
    if af >= 0:
        pygame.draw.rect(screen, BLUE, (cx, ay, int((aw // 2) * af), 22))
    else:
        wd = int((aw // 2) * (-af))
        pygame.draw.rect(screen, BLUE, (cx - wd, ay, wd, 22))
    screen.blit(font.render('STEER {:+6.1f}'.format(angle), True, WHITE), (ax, ay - 26))

    # 제목 + 조작 힌트
    screen.blit(big.render('teleop  (GTA style)', True, WHITE), (30, 14))
    hint = 'W accel  S brake  Q=4 E=10  A/D steer(hold)  SPACE stop  ESC quit'
    screen.blit(font.render(hint, True, GRAY), (30, H - 30))
    pygame.display.flip()


def main(args=None):
    # 노드 실행 진입점: pygame 초기화 -> 입력/주행 루프 -> 종료 시 정지 명령
    rclpy.init(args=args)
    node = TeleopKey()

    # pygame 창/폰트/시계 준비
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('teleop_key (focus me!)')
    font = pygame.font.SysFont(None, 24)
    big  = pygame.font.SysFont(None, 30)
    clock = pygame.time.Clock()

    held = set()
    running = True
    try:
        while running and rclpy.ok():
            dt = clock.tick(FPS) / 1000.0

            # 이벤트 처리: 키 눌림/뗌 집합 갱신 + 즉시 명령(ESC/SPACE/Q/E) + 포커스 상실 시 키 초기화
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    held.add(e.key)
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        node.speed = 0.0
                    elif e.key == pygame.K_q:
                        node.speed = 4.0
                    elif e.key == pygame.K_e:
                        node.speed = 10.0
                elif e.type == pygame.KEYUP:
                    held.discard(e.key)
                elif e.type == pygame.ACTIVEEVENT:
                    if getattr(e, 'gain', 1) == 0:
                        held.clear()

            focused = bool(pygame.key.get_focused())

            # 가속/브레이크 (둘 다 안 누르면 현재 속도 유지)
            if pygame.K_w in held:
                node.speed = min(MAX_SPEED, node.speed + ACCEL * dt)
            elif pygame.K_s in held:
                node.speed = max(MIN_SPEED, node.speed - BRAKE * dt)

            # 조향: A/D 누르고 있으면 서서히 증가, 떼면 서서히 중앙 복귀
            left  = pygame.K_a in held
            right = pygame.K_d in held
            if left and not right:
                node.angle = max(-MAX_ANGLE, node.angle - STEER_RATE * dt)
            elif right and not left:
                node.angle = min(MAX_ANGLE, node.angle + STEER_RATE * dt)
            else:
                node.angle = 0.0

            # 명령 발행 + 화면 갱신 + ROS 콜백 1회 처리
            node.publish()
            draw(screen, font, big, node.angle, node.speed, focused)
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 정지 명령을 한 번 보내고 자원 정리
        node.angle = 0.0
        node.speed = 0.0
        try:
            node.publish()
        except Exception:
            pass
        pygame.quit()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
