from track_drive.mux_node import limit_speed_increase


def test_lane_handoff_acceleration_is_limited():
    assert limit_speed_increase(15.0, 8.0, 0.1, 10.0) == 9.0


def test_lane_handoff_deceleration_is_immediate():
    assert limit_speed_increase(5.0, 8.0, 0.1, 10.0) == 5.0


def test_non_positive_rate_disables_limit():
    assert limit_speed_increase(15.0, 8.0, 0.1, 0.0) == 15.0
