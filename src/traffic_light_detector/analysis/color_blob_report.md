# ROI-first color-blob 구현 결과

## 변경 결과

검출 시작점을 어두운 외함 contour에서 점등 색상 blob으로 바꿨다. 입력 직후 고정 ROI
`(x=.30, y=.00, w=.40, h=.50)`를 자르고, 이후 HSV/mask/morphology/contour/구조 검사는
ROI 배열만 사용한다.

```text
ROI HSV → R/Y/G blob → blob geometry → equivalent-diameter pitch 1개
→ 4-slot bbox → distinct low-V components / darkness / contrast
→ relative slot state → 2-frame confirmation
```

금지된 고비용 경로와 full-frame mask는 사용하지 않는다.

## 기존 오검출 원인

기존 방식은 grayscale에서 어두운 가로 사각형을 먼저 찾았기 때문에 후보 생성 단계에
신호 색상이라는 의미 제약이 없었다. 천장 프레임, 선반, 의자, 그림자도 다음 조건을 쉽게
만족했다.

- 어둡고 가로로 긴 contour
- 신호등 외함과 비슷한 bounding-rect aspect
- bbox 내부의 우연한 소량 RED/YELLOW/GREEN pixel
- 전체적으로 어두워서 슬롯 darkness threshold를 만족하는 영역

특히 단순 low-V ratio는 균일하게 어두운 구조물도 “꺼진 슬롯 여러 개”로 해석한다. 기존
benchmark가 일부 negative 구간만 확인한 것도 rosbag 전체에서 반복되는 오류를 드러내지
못한 원인이다.

## 새 제거 조건

1. HSV 색 blob이 없으면 bbox 가설을 만들지 않는다.
2. blob의 area, width/height, aspect, circularity, mean S/V와 ROI 내 y 위치를 검사한다.
3. contour 등가 지름에서 pitch 하나만 계산한다. pitch를 최소값까지 강제로 확대하지 않고
   허용범위 밖이면 reject한다.
4. 색에 따라 RED=slot0, YELLOW=slot1, GREEN=slot2/3만 역산한다.
5. 예상 4-slot bbox가 ROI에 충분히 포함되지 않으면 reject한다.
6. 예상 비점등 슬롯에서 서로 다른 low-V connected component를 찾는다. 균일한 그림자처럼
   하나로 이어진 어두운 영역은 여기서 제거된다.
7. component 중심 간격과 예상 슬롯 중심 오차, component 면적과 크기 일관성을 검사한다.
8. 슬롯별 darkness 일관성, 외함/배경 V 대비, 예상 슬롯의 실제 color ratio를 추가 검사한다.
9. 통과 상태가 두 프레임 연속 확인돼야 filtered state가 바뀐다.

debug reject reason에는 `area`, `aspect_ratio`, `saturation`, `pitch`,
`bbox_outside_roi`, `dark_components`, `dark_component_size_cv`, `dark_slots`,
`housing_contrast`, `structure_score`, `slot_color_ratio` 등이 기록된다.

## 전체 rosbag benchmark

대상은 `bag_extract/20260806_130814`의 카메라 8,949프레임 전체다. 수동 검토한 상대시간
구간을 신호 ground truth로 쓰고, 그 밖의 프레임은 무신호로 계산했다.

| 항목 | 결과 |
|---|---:|
| 전체 frame | 8,949 |
| 실제 신호 frame | 415 |
| 실제 무신호 frame | 8,534 |
| true positive | 91 |
| false negative | 324 |
| false positive | 88 |
| true negative | 8,446 |
| precision | 50.8380% |
| recall | 21.9277% |
| false-positive frame rate | 1.0312% |
| 평균 처리시간 | 1.1305 ms |
| p99 처리시간 | 2.4009 ms |
| 평균시간 환산 core FPS | 884.6 FPS |

원거리 RED/YELLOW를 복구한 설정이다. 최종 filtered FP 88장 중 20장을 전체 bag에서
reservoir sampling해 저장했다. FN 20장과 상태오류 20장도 별도 저장했다. 수동 주석 종료
직후에도 화면상 신호가 남아 있는 프레임이 있어 일부 실제 신호가 FP로 계산되는 한계가 있다.

## recall 한계 분석

낮은 recall은 처리속도 문제가 아니다. 이 bag의 근접 RED/YELLOW 슬롯은 전체 영상 x=30%
왼쪽 또는 ROI 경계에 있고, LEFT 구간도 외함 오른쪽/왼쪽이 잘린다. 고정 ROI 밖 픽셀은
알고리즘이 볼 수 없고, 보이는 color fragment로 역산한 4구 bbox도 ROI 포함률 검사에서
reject된다. 요구사항의 고정 ROI와 false-positive 우선 정책을 지키면 이 손실은 불가피하다.

또한 수동 LEFT 구간은 실제로 LEFT와 GREEN이 함께 점등된 구간이다. 단일-state ground truth는
LEFT로 기록했다. LEFT 우선순위 적용 후 이 구간의 filtered LEFT는 41프레임, GREEN은
24프레임이다. 픽셀 손실 때문에 slot 2 점등 비율이
기준에 못 미친 프레임까지 복구하려면 ROI 또는 카메라 장착각 변경이 필요하다.

운영 전에 GUI의 `Original / ROI`에서 가장 가까운 신호등까지 4구 외함 전체가 고정 ROI 안에
들어오는지 확인해야 한다. ROI를 넓힐 수 없다면 카메라 장착각을 바꾸는 것이 recall 개선에
가장 직접적이다.
