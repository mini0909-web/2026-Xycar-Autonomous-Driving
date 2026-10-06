import pytest

from lane_drive.lane_drive_node import limit_speed_increase


def test_starts_at_configured_start_speed():
    assert limit_speed_increase(20.0, 0.0, 10.0, 8.0, 5.0) == 5.0


def test_limits_only_speed_increase():
    speed = limit_speed_increase(20.0, 5.0, 0.1, 8.0, 5.0)
    assert speed == pytest.approx(5.8)


def test_deceleration_is_immediate():
    assert limit_speed_increase(5.0, 20.0, 0.1, 8.0, 5.0) == 5.0


def test_non_positive_limit_disables_limiter():
    assert limit_speed_increase(20.0, 5.0, 0.1, 0.0, 5.0) == 20.0
