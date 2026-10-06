# 동적 장애물 회피

`dynamic_obstacle_avoidance`는 기존 라이다 클러스터링 ROI 안의 가장 가까운 전방
장애물을 연속 확인하고, 라이다 중앙선(`y=0`)을 기준으로 장애물 반대 방향에 고정
조향을 주는 ROS 2 노드입니다. 회피 중 장애물이 설정된 스캔 수만큼 연속으로
사라지면 `/dynamic/active=False`를 즉시 발행하며, `drive_mux`가 기본 E2E
`/motor_lane` 주행으로 복귀합니다. `drive_mux`는 `/xycar_motor`의 유일한 발행자입니다.

좌표계는 `x=전방`, `y=좌우`이고 양의 `y`가 왼쪽입니다. 차량 조향 규약은
`음수=좌회전`, `양수=우회전`이며 `steering_sign`으로 실차 방향을 보정할 수 있습니다.

## 토픽 계약

| 토픽 | 방향 | 역할 |
| --- | --- | --- |
| `/scan` | 입력 | `sensor_msgs/LaserScan` 라이다 입력 |
| `/motor_lane` | 입력 | 회피 해제 후 mux가 사용할 E2E 기본 명령 |
| `/obstacle/lane` | 출력 | 대상 방향. `1=LEFT`, `0=NONE`, `-1=RIGHT` |
| `/cmd/dynamic` | 출력 | 회피 중의 고정 `[steering, speed]` 명령 |
| `/dynamic/active` | 출력 | 회피 노드가 mux 제어권을 보유하는지 여부 |
| `/cone/active` | 입력 | 라바콘 회피 중 새 dynamic 회피 시작 억제 |
| `~/status` | 출력 | 상태, 대상 방향, 회피 방향, confirm/lost 카운트 |

다음 차선 토픽 구독과 관련 로직은 소스에 기존 코드가 주석 처리되어 있으며 현재
사용하지 않습니다.

- `/lane/ego_lane`
- `/lane/avoidance_ready`
- `/lane/handoff_ready`

`lane_drive`는 동적 회피의 판단 조건이 아니라 mux의 기본 E2E 주행 명령을 만들기
위해 launch에서 계속 실행됩니다.

## 판별과 상태 전환

기존 `_point_in_detection_roi()`와 클러스터링 범위는 그대로 사용합니다. 그 안에서
전방 거리, 전방 반폭, 각도, 최소 포인트 조건을 만족한 클러스터 중 가장 가까운 것을
대상으로 선택합니다.

- `target.y >= 0`: 장애물 `LEFT`, 오른쪽 회피 (`y=0` 포함)
- `target.y < 0`: 장애물 `RIGHT`, 왼쪽 회피

검출된 장애물은 라이다 정면 중앙선으로 즉시 양분하므로 미확정 방향, 이전 회피
방향 유지, 미확정으로 인한 정지 동작은 사용하지 않습니다.

상태 머신은 다음 두 상태만 사용합니다.

```text
E2E_FOLLOW --(장애물 연속 확인)--> AVOIDING
AVOIDING --(장애물 연속 미검출)--> E2E_FOLLOW
```

미검출 카운터는 제어 타이머가 아니라 새 `LaserScan` 콜백마다 한 번씩 증가합니다.
회피 중 다시 장애물을 검출하면 0으로 초기화합니다. 회피 방향은 회피가 끝날 때까지
고정되어 좌우 검출 노이즈에 따라 조향이 반전되지 않습니다.

## 주요 파라미터

| 파라미터 | 기본값 | 의미 |
| --- | ---: | --- |
| `detection_distance_m` | `1.8` | 기존 클러스터링 ROI 전방 범위 |
| `detection_half_width_m` | `0.5` | 기존 클러스터링 ROI 좌우 반폭 |
| `lidar_detection_half_angle_deg` | `110.0` | 기존 클러스터링 ROI 각도 |
| `front_detection_distance_m` | `1.3` | 실제 회피 대상 전방 거리 |
| `front_detection_half_width_m` | `0.3` | 실제 회피 대상 좌우 반폭 |
| `front_detection_min_points` | `3` | 대상 클러스터 최소 포인트 수 |
| `obstacle_confirm_scans` | `2` | 회피 시작 전 연속 검출 스캔 수 |
| `obstacle_lost_scans` | `3` | E2E 복귀 전 연속 미검출 스캔 수 |
| `avoidance_steering_angle` | `40.0` | E2E 각도를 더하지 않는 고정 회피각 |
| `avoidance_speed` | `7.0` | 회피 속도 |
| `max_steering` | `50.0` | 절대 조향 제한 |
| `steering_sign` | `1.0` | 실차 조향 방향 보정 부호 |

## MUX 동작

MUX 우선순위는 `CONE > DYNAMIC > LANE > STOP`입니다. MUX에는 차선 인식 결과로
dynamic 시작을 막는 조건이 없으므로 수정하지 않았습니다. dynamic active/command의
stale timeout과 dynamic에서 E2E로 돌아갈 때의 기존 가속 제한도 유지됩니다.

## 빌드와 실행

```bash
cd ~/ruby_ws
colcon build --packages-select dynamic_obstacle_avoidance track_drive
source install/setup.bash
ros2 launch dynamic_obstacle_avoidance dynamic_obstacle_avoidance.launch.py
```

실차에서는 차량을 들어 올리거나 낮은 `avoidance_speed`를 사용해 먼저 장애물 좌우별
조향 방향과 `steering_sign`을 확인하세요.
