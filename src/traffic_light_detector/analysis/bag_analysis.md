# 20260806_130814 rosbag 분석 및 검증 보고

## 1–5. 환경과 입력

- ROS 2: Humble
- 워크스페이스: `/home/user/dev/ros2/xycar_src`
- bag: `bag_extract/20260806_130814`, SQLite3, 7.7 GiB, 296.616787924 s,
  총 11,794 messages
- 토픽: `/scan` LaserScan 2,845개, `/usb_cam/image_raw/front`
  `sensor_msgs/msg/Image` 8,949개. `/clock`과 압축 토픽은 없음
- 영상: 640x480, `rgb8`, step 1920, frame_id `usb_cam`, 비압축
- FPS: 전체 span 기준 30.1669 Hz, timestamp median dt 0.033007 s;
  dt p01/p99 0.022132/0.043862 s
- 녹화 publisher QoS: reliable, volatile. 검출 subscriber는 reliable 및 best-effort
  publisher 모두와 연결되도록 best-effort/volatile depth 5 사용
- 같은 역할의 기존 OpenCV 신호등 패키지는 검색되지 않음

## 6. 대표 구간과 프레임

- RED(RED+YELLOW 동시 점등 포함): 80.0–82.0 s, 표본 81.523975 s
- YELLOW 단독: 82.1–82.8 s, 표본 82.050695 s
- LEFT+GREEN 동시: 약 138–143.5 s, 172–174 s, 210–212 s;
  LEFT 표본 142.540115 s
- GREEN 단독: 약 263–264.7 s, 표본 264.047865 s
- far: 140.000077 s, near: 82.513676 s
- 부분 잘림: 약 81.06 s 및 143.0 s, UNKNOWN 표본: 100.003425 s

`debug_samples/`에는 red/yellow/left/green/ceiling_light/far/near/unknown PNG가 있다.
ceiling_light 표본은 RED 프레임과 동일하므로 천장 조명과 신호를 동시에 비교할 수 있다.

## 7–8. 픽셀 분석과 포화 손실

아래 값은 원본 RGB를 BGR로 변환한 뒤 수동 측정한 center/ring ROI 결과다. 전체
mean/median/min/max/p05/p25/p75/p95는 `pixel_stats.json`에 보존했다.

| 영역 | BGR mean | HSV median | Lab mean | low-S/high-V | 완전 백색(각 채널≥250) | halo hue ratio |
|---|---|---|---|---:|---:|---:|
| RED center | 219.2/214.6/228.0 | 157/11/232 | 221.7/133.6/127.5 | 42.13% | 40.10% | red 12.69% |
| RED ring | 205.5/196.7/205.2 | 121/7/204 | 204.3/132.8/124.8 | 35.63% | 33.46% | red 8.46% |
| YELLOW center | 218.2/253.9/253.6 | 30/36/255 | 252.2/122.2/145.1 | 49.66% | 2.68% | yellow 51.68% |
| YELLOW ring | 151.9/171.9/185.0 | 30/44/203 | 177.9/130.3/140.0 | 25.55% | 6.09% | yellow 32.28% |
| GREEN center | 254.7/255.0/251.8 | 0/0/255 | 254.5/126.9/127.8 | 99.33% | 84.56% | green 2.01% |
| GREEN ring | 161.9/163.6/130.8 | 90/49/167 | 162.7/117.0/125.4 | 11.52% | 3.04% | green 70.54% |
| ceiling center | 246.0/244.2/243.5 | 0/0/255 | 245.2/128.1/127.0 | 84.78% | 84.19% | all 0% |
| ceiling edge | 184.4/176.6/175.6 | 106/14/179 | 183.6/129.2/123.9 | 15.38% | 14.25% | all 0% |
| housing | 164.1/165.6/162.7 | 84/8/170 | 172.8/126.6/128.6 | 0% | 0% | all 0% |

normalized RGB mean은 RED center `(R,G,B)=(0.345,0.323,0.330)`, YELLOW center
`(0.349,0.350,0.300)`, GREEN ring `(0.265,0.357,0.360)`, ceiling center
`(0.331,0.332,0.335)`였다. RED 중심은 40.1%, GREEN 중심은 84.6%가 완전히 흰색이라
그 픽셀만으로 원 Hue를 복구할 수 없다. YELLOW는 R/G channel clipping이 94% 이상이지만
B가 낮고 Lab-b와 ring Hue가 남았다. 따라서 HSV 단독 복구는 불가능하며 ring/외함/슬롯
구조가 필수다.

## 9. 생성 파일

- `detector_core.py`: 마스크, 후보, 외함/halo/white score, tracking, temporal filter
- `traffic_light_detector_node.py`: Image/CompressedImage 구독과 ROS 출력
- `hsv_tuner_node.py`: 공용 코어 기반 HighGUI 튜너와 YAML 저장/불러오기
- `config/traffic_light.yaml`, launch 2개, package metadata
- `test/test_detector_core.py`, `test/test_tuner_config.py`
- `tools/analyze_bag.py`, `tools/monitor_topics.py`
- `debug_samples/*.png`, `analysis/pixel_stats.json`, 이 보고서와 README

## 10–14. 검출, 천장 제거, halo, tracking, 슬롯

점등 contour를 즉시 상태로 만들지 않고 가능한 슬롯과 pitch scale을 순회해 전체 4구
bbox를 생성한다. Hough 원 일치, 네 슬롯 ring edge/darkness, 저채도 중간 회색 외함,
bbox aspect, 후보 위치와 시간 연속성을 조합한다. SEARCHING 후보가 raw state를 3프레임
확인하기 전에는 tracker를 시작하지 않는다.

천장 조명은 중심/edge 모두 hue halo가 0이고, 길쭉한 contour aspect에서 먼저 제거되거나
4개의 어두운 원형 슬롯 및 중간 회색 외함 비율에서 `NO_HOUSING`이 된다. 중심과 ring이
모두 저채도 고휘도이고 halo/구조/시간 점수가 없을수록 `white_light_score`가 커진다.
완전 백색이며 구조도 없으면 RED/YELLOW가 아니라 UNKNOWN이다.

CSRT를 먼저 생성하고 실패하면 KCF, 둘 다 없으면 주기적 재검출을 사용한다. 5프레임마다
실제 구조를 재검출하고 중심 jump 0.40, 크기 jump 0.60을 넘는 후보를
`LOW_TEMPORAL_CONFIDENCE`로 제거한다. 실패 5프레임까지 LOST, 이후 SEARCHING이다.
입력 timestamp가 역행하거나 1 s 넘게 끊기면 상태/트래커를 초기화한다.

검출 bbox를 `[0,.25,.50,.75,1]`로 나누고 각 슬롯에 10% margin을 둔다. 같은 green mask를
쓰는 LEFT/GREEN은 슬롯 2/3의 상대 x로 구분하며, 동시 점등 점수가 비슷할 때 LEFT를 우선한다.

## 15–19. 최종 파라미터

- HSV: RED1 H 0–12/S 55–255/V 75–255, RED2 H 168–179/S 55–255/V 75–255,
  YELLOW H 12–38/S 45–255/V 90–255, GREEN H 38–102/S 35–255/V 70–255
- 보조: red normalized-R≥0.39, R-max(G,B)≥18, Lab-a≥136;
  yellow normalized R+G≥0.69, normalized-B≤0.25, min(R,G)-B≥18, Lab-b≥142;
  green normalized-G≥0.36, G-R≥8, Lab-a≤128
- white: V≥238, S≤48; halo minimum 0.045, dilation 9
- 신호 가중치: pixel .20, halo .25, brightness .10, area .15, circularity .10,
  slot position .10, housing .10; min score .43, margin .055
- bbox: min 64x18, aspect 2.7–5.8; housing min mid-gray ratio .38/confidence .88
- tracking: CSRT, redetect 5, max lost 5, expand 1.8, smoothing alpha .40,
  center/size jump .40/.60
- temporal: confirm 3, history 5, UNKNOWN hold 3, timeout .5 s

## 20–22. GUI와 YAML

사용법과 키는 `README.md`의 HSV 튜너 절을 따른다. detector와 tuner는 동일한
`TrafficLightDetectorCore`를 호출한다. `s`가 detector 형식 YAML을 저장하고 `l`이 다시
읽는다. 저장/불러오기 round-trip은 단위 테스트로 확인했다. 이 작업 환경에는 DISPLAY=:1이
있었지만 최종 자동 검증은 비대화형 세션이라 실제 사람의 trackbar 조작 확인은 하지 않았다.

## 23–26. 빌드와 검증

- 패키지 제한 colcon build 성공. 전체 워크스페이스 탐색은 기존 여러 패키지의 비표준
  `setup.py` 때문에 실패하므로 `--base-paths traffic_light_detector`가 필요하다.
- pytest: 13 passed.
- ROS2 실제 bag 저속 스트리밍에서 `/traffic_light/state`, `/traffic_light/scores`, debug/mask
  토픽 수신 확인. RED 5회, YELLOW 1회를 포함했고 score 최대는
  `[.8777,.6651,.4435,.2590,.7877,.4411]`이었다.
- 최종 코어 대표 frame: RED score `.804/.718/.199/.239`, detection `.750`;
  YELLOW `.227/.776/.269/.161`, detection `.820`;
  LEFT `.004/.007/.899/.884`, detection `.808`;
  GREEN `.000/.000/.738/.906`, detection `.805`.
- 연속 offline 검증: 141–145 s에서 LEFT 69, GREEN 4, UNKNOWN 48 frame이며 통과 뒤에는
  UNKNOWN으로 복귀. 신호가 없고 천장 조명이 보이는 90–95 s의 152/152 frame은 UNKNOWN;
  이 구간 천장 조명의 RED/YELLOW 오검출은 각각 0회였다.
- fresh SEARCHING에서 near 표본은 진입 위치 제한 때문에 UNKNOWN이며, 실제 연속 재생에서는
  멀리서 획득한 bbox를 tracking하여 near까지 내려온다.
- 비대화형 DDS 재검증의 처리 출력은 약 1–11 Hz로 원본 30 Hz보다 낮았다. 디버그 JPEG 2개,
  Python Hough/CSRT 비용이 주 원인이라 “영상 FPS를 따라감” 항목은 미충족이다.

## 27. 명령

```bash
cd /home/user/dev/ros2/xycar_src
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths traffic_light_detector \
  --packages-select traffic_light_detector
source install/setup.bash
ros2 launch traffic_light_detector traffic_light_detector.launch.py
ros2 bag play bag_extract/20260806_130814 --loop
ros2 topic echo /traffic_light/state
ros2 topic echo /traffic_light/scores
ros2 topic hz /traffic_light/debug/compressed
ros2 launch traffic_light_detector hsv_tuner.launch.py
```

## 28–29. 남은 문제와 개선

- CPU-only Python에서 30 Hz를 만족하지 못한다. Hough를 C++ 노드로 옮기거나 contour 기반
  원 검증을 vectorize하고, 디버그 JPEG rate를 낮추며 MultiThreadedExecutor를 쓰는 개선이 필요하다.
- Search acquisition 위치 제한은 이 bag의 진입 궤적에 맞춘 값이다. 카메라/코스가 바뀌면 GUI로
  acquisition 범위와 min bbox를 다시 측정해야 한다.
- 동시 RED+YELLOW는 score margin으로 RED, LEFT+GREEN은 LEFT 우선 정책이다. 차량 제어 규칙이
  다른 경우 다중 상태 메시지 또는 명시적 우선순위 파라미터를 추가해야 한다.
- 비대화형 세션에서는 실제 trackbar 드래그를 자동화하지 않았다. GUI 창/키/저장 함수는 구현됐고
  YAML round-trip은 검증됐지만, 운영 PC에서 한 번 시각 확인하는 것이 좋다.
