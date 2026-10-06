# traffic_light_detector

ROS 2 Humble용 OpenCV 기반 4구 신호등 검출기입니다. 딥러닝 모델을 사용하지 않으며,
`[RED] [YELLOW] [LEFT] [GREEN]` 배열을 색상, 과노출 중심 주변 halo, 회색 외함,
네 개의 원형 슬롯, 상대 위치, CSRT/KCF 추적과 시간 필터로 판별합니다.

## 실행

이 워크스페이스에는 별개의 기존 Python 패키지 중 colcon 탐색이 실패하는 패키지가 있으므로
새 패키지만 빌드할 때는 `--base-paths`를 함께 사용합니다.

```bash
cd /home/user/dev/ros2/xycar_src
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --base-paths traffic_light_detector \
  --packages-select traffic_light_detector
source install/setup.bash

ros2 launch traffic_light_detector traffic_light_detector.launch.py
ros2 bag play bag_extract/20260806_130814 --loop
```

다른 설정 파일은 다음처럼 지정합니다.

```bash
ros2 launch traffic_light_detector traffic_light_detector.launch.py \
  params_file:=/absolute/path/to/traffic_light_tuned.yaml
```

입력이 `sensor_msgs/msg/CompressedImage`이면 YAML에서 `image_topic`과
`input_compressed: true`를 지정합니다. detector는 GUI 없이 독립 실행되며
`gui.enabled` 값과 무관하게 HighGUI 창을 만들지 않습니다.

## 출력

- `/traffic_light/state` (`std_msgs/msg/String`): `RED`, `YELLOW`, `LEFT`, `GREEN`, `UNKNOWN`
- `/traffic_light/scores` (`std_msgs/msg/Float32MultiArray`):
  `[red, yellow, left, green, detection_confidence, white_light_score]`
- `/traffic_light/debug/compressed`: 검출/추적/점수 오버레이 JPEG
- `/traffic_light/mask/compressed`: 공용 후보 morphology mask JPEG

디버그 오버레이는 점등 색과 4구 슬롯 위치가 일치하는 후보만 최종 박스로
선택합니다. 기본 설정에서는 3프레임 연속 일치해 확정된 박스만 표시하며,
천장등/반사광 같은 거절 후보 표시는 숨겨집니다. 원인 분석이 필요할 때만
다음처럼 다시 켤 수 있습니다.

```bash
ros2 param set /traffic_light_detector debug.show_rejected_candidates true
```

제공된 실내 코스의 기본 Search ROI는 영상 높이의 `24%~54%` 구간입니다.
신규 후보 중심은 `24%~38%`, x는 `25%~66%`로 제한하며, 확인된 신호의
tracker만 ROI 하단 `54%`까지 이동할 수 있습니다.
확정 신호가 ROI 밖으로 빠져 추적이 끝나면 4초간 재획득을 막아, 지나간
신호등 주변의 의자·사람·바닥 반사에 다시 붙지 않도록 합니다.

## 알고리즘

넓은 Search ROI 안에서 RED/YELLOW/GREEN 및 고휘도 저채도 중심을 찾습니다. 후보마다
점등 contour가 어느 슬롯에 있는지를 가정해 전체 4구 bbox를 만들고 다음을 검증합니다.

1. 네 예상 슬롯에서 원형 림(Hough/edge) 일치
2. 저채도 중간 회색 외함과 어두운 림 비율
3. 중심 disk와 팽창 ring을 분리한 RED/YELLOW/GREEN halo 비율
4. contour 원형도와 bbox 종횡비
5. 이전 bbox IoU, 중심/크기 jump, 화면 위치
6. 중심과 ring이 모두 희고 halo/외함이 없을 때 `white_light_score` 가산

SEARCHING 중에는 실제 bag에서 측정한 진입 영역을 사용해 천장/바닥 후보가 tracker를
선점하지 않게 합니다. 신호가 시간적으로 확인된 뒤에만 TRACKING으로 전환합니다.
TRACKING은 CSRT, KCF, 재검출 fallback 순서이며 매 `redetect_interval` 프레임마다 실제
구조 검출로 보정합니다. 점수가 낮거나 비슷하고 구조도 확실하지 않으면 UNKNOWN입니다.

## HSV 튜너

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch traffic_light_detector hsv_tuner.launch.py
```

창: `Controls`, `Original`, `HSV Mask`. `Original`에는 Search ROI와 검출 bbox가
표시됩니다. Original을 클릭하면 BGR/HSV/Lab/normalized RGB, 슬롯과 ROI 포함 여부가
표시됩니다. 드래그하면 영역 평균·중앙값·5/95 percentile을 로그로 출력합니다.

키:

- `1`/`2`/`3`/`4`: RED1/RED2/YELLOW/GREEN 선택
- `s`: `gui.config_save_path`에 현재 설정 저장
- `l`: 해당 YAML 다시 불러오기
- `r`: 코어 기본값으로 초기화
- `p`: 현재 설정 출력
- `space`: 일시정지/재생
- `n`: 일시정지 상태에서 다음 수신 프레임
- `q` 또는 `ESC`: 종료

H/S/V min/max는 내부에서 정렬·clamp되며 morphology kernel은 1 이상의 홀수로
보정됩니다. 설치 share가 읽기 전용일 수 있으므로 저장 경로는 반드시 쓰기 가능한
절대 경로로 설정하십시오. 저장 YAML은 detector의 `params_file`로 바로 재사용합니다.
DISPLAY가 없으면 튜너는 명확한 오류를 기록하고 창을 만들지 않으며 detector에는 영향이 없습니다.

## 재현 가능한 분석과 테스트

```bash
source /opt/ros/humble/setup.bash
python3 traffic_light_detector/tools/analyze_bag.py \
  bag_extract/20260806_130814

PYTHONPATH=traffic_light_detector:$PYTHONPATH \
  python3 -m pytest -q traffic_light_detector/test
```

원시 통계는 `analysis/pixel_stats.json`, 대표 영상은 `debug_samples/`에 있습니다.
bag별 상세 결과와 한계는 `analysis/bag_analysis.md`를 참고하십시오.
