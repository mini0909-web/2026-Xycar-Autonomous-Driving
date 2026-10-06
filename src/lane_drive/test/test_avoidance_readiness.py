import numpy as np

from lane_drive.lane_drive_node import (
    LaneResult,
    lane_ready_for_avoidance,
    lane_ready_for_handoff,
)


def make_result(
    *, valid=True, center=320.0, reason='both lanes', yellow_side=1,
    left_visible=True, right_visible=True,
):
    return LaneResult(
        valid=valid,
        lane_center_x=center,
        left_fit=np.array([0.0]) if left_visible else None,
        right_fit=np.array([0.0]) if right_visible else None,
        mask=np.zeros((1, 1), dtype=np.uint8),
        lookahead_y=0,
        confidence=1.0,
        reason=reason,
        yellow_side=yellow_side,
        yellow_confidence=1.0,
    )


def test_ready_requires_yellow_divider_one_lane_and_centered_vehicle():
    assert lane_ready_for_avoidance(make_result(), 640, 0.0, 60.0)
    assert lane_ready_for_avoidance(
        make_result(reason='single lane', right_visible=False), 640, 0.0, 60.0
    )
    assert not lane_ready_for_avoidance(make_result(center=390.0), 640, 0.0, 60.0)
    assert not lane_ready_for_avoidance(
        make_result(left_visible=False, right_visible=False), 640, 0.0, 60.0
    )
    assert not lane_ready_for_avoidance(make_result(yellow_side=0), 640, 0.0, 60.0)


def test_handoff_requires_both_lanes_and_tighter_centering():
    assert lane_ready_for_handoff(make_result(center=350.0), 640, 0.0, 40.0)
    assert not lane_ready_for_handoff(make_result(center=370.0), 640, 0.0, 40.0)
    assert not lane_ready_for_handoff(
        make_result(right_visible=False), 640, 0.0, 40.0
    )
