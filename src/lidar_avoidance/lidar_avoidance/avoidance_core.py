"""Pure geometry helpers used by the LiDAR avoidance ROS node."""

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class ZoneSummary:
    """Number of LiDAR returns and nearest range inside one rectangular zone."""

    point_count: int
    nearest_range: float


def scan_to_points(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    y_sign: float = 1.0,
) -> List[Point2D]:
    """Convert valid LaserScan samples into vehicle-frame Cartesian points."""
    points: List[Point2D] = []
    minimum = max(0.0, float(range_min))
    maximum = max(minimum, float(range_max))
    y_multiplier = 1.0 if y_sign >= 0.0 else -1.0

    for index, raw_range in enumerate(ranges):
        distance = float(raw_range)
        if not math.isfinite(distance) or not (minimum <= distance <= maximum):
            continue
        angle = float(angle_min) + index * float(angle_increment)
        points.append(
            (
                distance * math.cos(angle),
                y_multiplier * distance * math.sin(angle),
            )
        )
    return points


def summarize_box(
    points: Iterable[Point2D],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> ZoneSummary:
    """Summarize points inside an axis-aligned vehicle-frame rectangle."""
    count = 0
    nearest = math.inf
    for x, y in points:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            count += 1
            nearest = min(nearest, math.hypot(x, y))
    return ZoneSummary(count, nearest)


def lane_change_bias(
    elapsed_sec: float,
    duration_sec: float,
    steering_direction: float,
    maximum_bias: float,
) -> float:
    """Return a smooth S-curve steering bias that starts and finishes at zero."""
    duration = max(1e-6, float(duration_sec))
    phase = min(1.0, max(0.0, float(elapsed_sec) / duration))
    return (
        float(steering_direction)
        * abs(float(maximum_bias))
        * math.sin(2.0 * math.pi * phase)
    )
