import cv2
import numpy as np
import pytest

from traffic_light_detector.detector_core import (
    Candidate,
    TemporalStateFilter,
    TrafficLightDetectorCore,
    bbox_iou,
    clip_bbox,
    sanitized_hsv,
    slot_boxes,
)


def synthetic_signal(slot, colour):
    image = np.full((240, 420, 3), 35, np.uint8)
    bbox = (50, 65, 320, 82)
    cv2.rectangle(image, (50, 65), (370, 147), (165, 165, 165), -1)
    centers = [(90 + index * 80, 106) for index in range(4)]
    for center in centers:
        cv2.circle(image, center, 29, (35, 35, 35), -1)
        cv2.circle(image, center, 23, (125, 125, 125), -1)
    center = centers[slot]
    cv2.circle(image, center, 22, colour, -1)
    cv2.circle(image, center, 10, (255, 255, 255), -1)
    return image, bbox


def test_bbox_clipping_and_iou():
    assert clip_bbox((-10, -5, 30, 20), (100, 100, 3)) == (0, 0, 20, 15)
    assert clip_bbox((5, 5, -1, 3), (100, 100, 3)) is None
    assert bbox_iou((0, 0, 10, 10), (5, 5, 10, 10)) == pytest.approx(25 / 175)


def test_four_slot_split_and_margin():
    boxes = slot_boxes((0, 0, 400, 80), [0, .25, .5, .75, 1], .1, (100, 500))
    assert len(boxes) == 4
    assert boxes[0] == (10, 8, 80, 64)
    assert boxes[3] == (310, 8, 80, 64)


def test_invalid_hsv_is_ordered_and_clamped():
    low, high = sanitized_hsv({"hsv.red1.h_min": 190, "hsv.red1.h_max": -3,
                               "hsv.red1.s_min": 260, "hsv.red1.s_max": 10,
                               "hsv.red1.v_min": 250, "hsv.red1.v_max": 20}, "red1")
    assert low.tolist() == [0, 10, 20]
    assert high.tolist() == [179, 255, 250]


@pytest.mark.parametrize("slot,colour,expected", [
    (0, (0, 0, 255), "RED"),
    (1, (0, 255, 255), "YELLOW"),
    (2, (0, 255, 80), "LEFT"),
    (3, (0, 255, 80), "GREEN"),
])
def test_slot_classification_with_saturated_core_and_halo(slot, colour, expected):
    image, bbox = synthetic_signal(slot, colour)
    core = TrafficLightDetectorCore({"temporal_filter.confirm_frames": 1})
    masks = core.build_masks(image, include_circles=False)
    structure, housing, _, _ = core._structure(masks, bbox)
    candidate = Candidate(bbox, (0, 0), slot, .9, .1, max(.8, structure),
                          max(.8, housing), {"red": .2, "yellow": .2, "green": .2}, .8, .5)
    state, scores, _ = core._classify(masks, candidate)
    assert state == expected
    assert scores[expected] >= core.config["score.min_score"]


def test_course_left_requires_simultaneous_straight_green():
    image, bbox = synthetic_signal(2, (0, 255, 80))
    core = TrafficLightDetectorCore({
        "score.left_simultaneous_green_min": .55,
        "score.left_pair.left_min_pixel_ratio": .15,
        "score.left_pair.left_min_area_ratio": .20,
        "score.left_pair.left_min_brightness": .90,
        "score.left_pair.green_min_pixel_ratio": .12,
        "score.left_pair.green_min_area_ratio": .25,
        "score.left_pair.green_min_brightness": .95,
    })
    masks = core.build_masks(image, include_circles=False)
    structure, housing, _, _ = core._structure(masks, bbox)
    candidate = Candidate(bbox, (0, 0), 2, .9, .1, max(.8, structure),
                          max(.8, housing), {"red": .2, "yellow": .2, "green": .2}, .8, .5)
    state, _, _ = core._classify(masks, candidate)
    assert state == "UNKNOWN"

    # This course's protected-left phase illuminates slot 2 and slot 3 together.
    cv2.circle(image, (330, 106), 22, (0, 255, 80), -1)
    cv2.circle(image, (330, 106), 10, (255, 255, 255), -1)
    masks = core.build_masks(image, include_circles=False)
    state, scores, _ = core._classify(masks, candidate)
    assert state == "LEFT"
    assert scores["GREEN"] >= core.config["score.left_simultaneous_green_min"]


def test_white_blob_without_housing_is_rejected():
    image = np.full((240, 420, 3), 40, np.uint8)
    cv2.circle(image, (180, 80), 18, (255, 255, 255), -1)
    core = TrafficLightDetectorCore({"tracking.enabled": False,
                                     "temporal_filter.confirm_frames": 1})
    result = core.detect(image, 0.0, False)
    assert result.raw_state == "UNKNOWN"
    assert result.bbox is None
    assert all(candidate.rejected for candidate in result.candidates)


def test_slot_consistent_candidate_is_selected_over_higher_wrong_alignment():
    image, expected_bbox = synthetic_signal(0, (0, 0, 255))
    core = TrafficLightDetectorCore({"tracking.enabled": False,
                                     "search_roi.y": 0.0,
                                     "search_roi.h": 0.60,
                                     "candidate.acquire_max_center_y_ratio": 0.60,
                                     "temporal_filter.confirm_frames": 1})
    result = core.detect(image, 0.0, False)
    assert result.raw_state == "RED"
    assert result.bbox is not None
    assert bbox_iou(result.bbox, expected_bbox) > 0.55


def test_unconfirmed_signal_hides_proposal_until_confirmation():
    image, _ = synthetic_signal(0, (0, 0, 255))
    core = TrafficLightDetectorCore({"tracking.enabled": False,
                                     "search_roi.y": 0.0,
                                     "search_roi.h": 0.60,
                                     "candidate.acquire_max_center_y_ratio": 0.60,
                                     "temporal_filter.confirm_frames": 3})
    first = core.detect(image, 0.0, False)
    second = core.detect(image, 0.1, False)
    result = core.detect(image, 0.2, False)
    assert first.proposal_bbox is None
    assert second.proposal_bbox is None
    assert result.state == "RED"
    assert result.bbox is not None
    assert result.proposal_state == "RED"
    assert result.proposal_bbox is not None


def test_candidate_nms_removes_duplicate_slot_boxes():
    image, _ = synthetic_signal(0, (0, 0, 255))
    core = TrafficLightDetectorCore({"candidate.nms_iou_threshold": 0.90})
    masks = core.build_masks(image)
    candidates = core._hypotheses(image, masks)
    for index, candidate in enumerate(candidates):
        assert all(candidate.assigned_slot != other.assigned_slot or
                   bbox_iou(candidate.bbox, other.bbox) < 0.90
                   for other in candidates[index + 1:])


def test_accepted_candidates_have_distinct_circle_slot_support():
    image, _ = synthetic_signal(0, (0, 0, 255))
    core = TrafficLightDetectorCore({"search_roi.y": 0.0,
                                     "search_roi.h": 0.60,
                                     "candidate.acquire_max_center_y_ratio": 0.60})
    candidates = core._hypotheses(image, core.build_masks(image))
    accepted = [candidate for candidate in candidates if not candidate.rejected]
    assert accepted
    assert all(candidate.circle_slots >= core.config["housing.min_circle_slots"]
               for candidate in accepted)


def test_simultaneous_left_and_green_prefers_left_state():
    image, _ = synthetic_signal(2, (0, 255, 80))
    cv2.circle(image, (330, 106), 22, (0, 255, 80), -1)
    cv2.circle(image, (330, 106), 10, (255, 255, 255), -1)
    core = TrafficLightDetectorCore({"tracking.enabled": False,
                                     "search_roi.y": 0.0,
                                     "search_roi.h": 0.60,
                                     "candidate.acquire_max_center_y_ratio": 0.60,
                                     "temporal_filter.confirm_frames": 1})
    result = core.detect(image, 0.0, False)
    assert result.raw_state == "LEFT"


def test_white_light_penalty_lowers_red_score():
    image, bbox = synthetic_signal(0, (255, 255, 255))
    core = TrafficLightDetectorCore()
    masks = core.build_masks(image, include_circles=False)
    low, _ = core._slot_feature(masks, bbox, 0, "RED", .8, .8, 1.0)
    core.update_config({"score.white_penalty_weight": 0.0})
    unpenalized, _ = core._slot_feature(masks, bbox, 0, "RED", .8, .8, 1.0)
    assert low < unpenalized


def test_low_and_ambiguous_scores_become_unknown():
    core = TrafficLightDetectorCore({"score.min_score": .99})
    image, bbox = synthetic_signal(0, (0, 0, 255))
    masks = core.build_masks(image, include_circles=False)
    candidate = Candidate(bbox, (0, 0), 0, .5, .2, .5, .5, {}, .5, .5)
    assert core._classify(masks, candidate)[0] == "UNKNOWN"


def test_temporal_confirmation_and_bounded_unknown_hold():
    filt = TemporalStateFilter(confirm_frames=2, history_size=4,
                               unknown_hold_frames=2, hold_timeout=.5)
    assert filt.update("RED", 0.0) == "UNKNOWN"
    assert filt.update("RED", 0.1) == "RED"
    assert filt.update("UNKNOWN", 0.2) == "RED"
    assert filt.update("UNKNOWN", 0.3) == "RED"
    assert filt.update("UNKNOWN", 0.4) == "UNKNOWN"
    assert filt.update("RED", 1.0) == "UNKNOWN"
    assert filt.update("RED", 1.1) == "RED"
    assert filt.update("UNKNOWN", 2.0) == "UNKNOWN"


def test_left_uses_stricter_temporal_confirmation():
    filt = TemporalStateFilter(confirm_frames=2, history_size=3,
                               left_confirm_frames=5)
    for index in range(4):
        assert filt.update("LEFT", index * .1) == "UNKNOWN"
    assert filt.update("LEFT", .4) == "LEFT"


def test_tracking_lost_then_searching():
    core = TrafficLightDetectorCore({"tracking.enabled": False,
                                     "tracking.max_lost_frames": 2,
                                     "temporal_filter.confirm_frames": 1})
    core.previous_bbox = (50, 65, 320, 82)
    core.tracking_state = "TRACKING"
    blank = np.zeros((240, 420, 3), np.uint8)
    first = core.detect(blank, .1, False)
    second = core.detect(blank, .2, False)
    third = core.detect(blank, .3, False)
    assert first.tracking_state == "LOST"
    assert second.tracking_state == "LOST"
    assert third.tracking_state == "SEARCHING"


def test_confirmed_track_loss_starts_reacquisition_cooldown():
    signal, _ = synthetic_signal(0, (0, 0, 255))
    blank = np.zeros_like(signal)
    core = TrafficLightDetectorCore({"tracking.enabled": False,
                                     "tracking.max_lost_frames": 1,
                                     "tracking.reacquire_cooldown_sec": 4.0,
                                     "search_roi.y": 0.0,
                                     "search_roi.h": 0.60,
                                     "candidate.acquire_max_center_y_ratio": 0.60,
                                     "temporal_filter.confirm_frames": 1})
    assert core.detect(signal, 0.0, False).state == "RED"
    core.detect(blank, 0.1, False)
    assert core.detect(blank, 0.2, False).tracking_state == "SEARCHING"
    blocked = core.detect(signal, 0.3, False)
    assert blocked.state == "UNKNOWN"
    assert blocked.bbox is None
    assert core.detect(signal, 4.3, False).state == "RED"
