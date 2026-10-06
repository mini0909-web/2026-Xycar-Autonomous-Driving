import math

from lidar_avoidance.avoidance_core import (
    lane_change_bias,
    scan_to_points,
    summarize_box,
)
import pytest


def test_scan_to_points_uses_laserscan_angles_and_filters_invalid_ranges():
    points = scan_to_points(
        [1.0, 0.0, math.inf, 2.0],
        angle_min=0.0,
        angle_increment=math.pi / 2.0,
        range_min=0.1,
        range_max=3.0,
    )

    assert len(points) == 2
    assert points[0] == pytest.approx((1.0, 0.0))
    assert points[1] == pytest.approx((0.0, -2.0), abs=1e-6)


def test_summarize_box_counts_only_points_inside_zone():
    summary = summarize_box(
        [(0.5, 0.0), (1.0, 0.1), (0.5, 0.5), (-0.2, 0.0)],
        0.2,
        1.2,
        -0.2,
        0.2,
    )

    assert summary.point_count == 2
    assert summary.nearest_range == pytest.approx(0.5)


def test_lane_change_bias_is_s_curve_and_finishes_at_zero():
    assert lane_change_bias(0.0, 2.0, -1.0, 20.0) == pytest.approx(0.0)
    assert lane_change_bias(0.5, 2.0, -1.0, 20.0) == pytest.approx(-20.0)
    assert lane_change_bias(1.5, 2.0, -1.0, 20.0) == pytest.approx(20.0)
    assert lane_change_bias(2.0, 2.0, -1.0, 20.0) == pytest.approx(0.0)
