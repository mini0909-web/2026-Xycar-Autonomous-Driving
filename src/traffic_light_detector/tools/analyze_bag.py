#!/usr/bin/env python3
"""Extract reproducible bag samples and write colour-space statistics.

Run after sourcing ROS 2 Humble, from the package source directory.  The sample
times and pixel regions were measured on the supplied 640x480 bag.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


SAMPLES = {
    "red": 81.52,
    "yellow": 82.05,
    "left": 142.54,
    "green": 264.04,
    "ceiling_light": 81.52,
    "far": 140.00,
    "near": 82.51,
    "unknown": 100.00,
}

# kind, sample, geometry. Circle=(cx,cy,r), ring=(cx,cy,r1,r2), rect=(x1,y1,x2,y2)
REGIONS = {
    "red_center": ("red", "circle", (132, 128, 8)),
    "red_halo": ("red", "ring", (132, 128, 10, 23)),
    "yellow_center": ("yellow", "circle", (193, 136, 7)),
    "yellow_halo": ("yellow", "ring", (193, 136, 9, 22)),
    "green_center": ("green", "circle", (375, 73, 7)),
    "green_halo": ("green", "ring", (375, 73, 9, 21)),
    "ceiling_center": ("ceiling_light", "rect", (281, 50, 291, 95)),
    "ceiling_edge": ("ceiling_light", "rect_ring", (273, 38, 299, 108, 281, 50, 291, 95)),
    "housing": ("red", "rect", (216, 100, 235, 116)),
    "background": ("red", "rect", (370, 235, 440, 305)),
}


def region_mask(shape, kind, geometry):
    mask = np.zeros(shape[:2], np.uint8)
    if kind == "circle":
        cv2.circle(mask, geometry[:2], geometry[2], 255, -1)
    elif kind == "ring":
        cv2.circle(mask, geometry[:2], geometry[3], 255, -1)
        cv2.circle(mask, geometry[:2], geometry[2], 0, -1)
    elif kind == "rect":
        cv2.rectangle(mask, geometry[:2], geometry[2:], 255, -1)
    else:
        cv2.rectangle(mask, geometry[:2], geometry[2:4], 255, -1)
        cv2.rectangle(mask, geometry[4:6], geometry[6:8], 0, -1)
    return mask.astype(bool)


def triples(values):
    flat = values.reshape(-1, 3).astype(float)
    return {
        "mean": np.mean(flat, axis=0).round(3).tolist(),
        "median": np.median(flat, axis=0).round(3).tolist(),
        "min": np.min(flat, axis=0).round(3).tolist(),
        "max": np.max(flat, axis=0).round(3).tolist(),
        "p05": np.percentile(flat, 5, axis=0).round(3).tolist(),
        "p25": np.percentile(flat, 25, axis=0).round(3).tolist(),
        "p75": np.percentile(flat, 75, axis=0).round(3).tolist(),
        "p95": np.percentile(flat, 95, axis=0).round(3).tolist(),
    }


def statistics(frame, selector):
    bgr = frame[selector]
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV); hsv = hsv_image[selector]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[selector]
    floating = bgr.astype(float); total = np.sum(floating, axis=1, keepdims=True) + 1.0
    normalized_bgr = floating / total
    hue = hsv[:, 0]
    red = ((hue <= 12) | (hue >= 168)) & (hsv[:, 1] >= 45)
    yellow = (hue >= 12) & (hue <= 38) & (hsv[:, 1] >= 35)
    green = (hue >= 38) & (hue <= 102) & (hsv[:, 1] >= 30)
    return {
        "pixel_count": int(len(bgr)),
        "BGR": triples(bgr), "HSV": triples(hsv), "Lab": triples(lab),
        "normalized_RGB_mean": np.mean(normalized_bgr[:, ::-1], axis=0).round(5).tolist(),
        "normalized_RGB_median": np.median(normalized_bgr[:, ::-1], axis=0).round(5).tolist(),
        "low_s_high_v_ratio": round(float(np.mean((hsv[:, 1] <= 35) & (hsv[:, 2] >= 245))), 6),
        "fully_white_250_ratio": round(float(np.mean(np.all(bgr >= 250, axis=1))), 6),
        "any_channel_250_ratio": round(float(np.mean(np.any(bgr >= 250, axis=1))), 6),
        "channel_250_ratio_BGR": np.mean(bgr >= 250, axis=0).round(6).tolist(),
        "halo_colour_ratios": {"red": round(float(np.mean(red)), 6),
                               "yellow": round(float(np.mean(yellow)), 6),
                               "green": round(float(np.mean(green)), 6)},
    }


def extract(bag, output):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    topic = "/usb_cam/image_raw/front"; message_type = get_message(types[topic])
    pending = dict(SAMPLES); frames = {}; actual_times = {}; start = None
    while reader.has_next() and pending:
        read_topic, data, timestamp = reader.read_next()
        if read_topic != topic:
            continue
        if start is None:
            start = timestamp
        relative = (timestamp - start) / 1e9
        due = [name for name, target in pending.items() if relative >= target]
        if not due:
            continue
        message = deserialize_message(data, message_type)
        rgb = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step // 3, 3)
        bgr = cv2.cvtColor(rgb[:, :message.width], cv2.COLOR_RGB2BGR)
        for name in due:
            frames[name] = bgr.copy(); actual_times[name] = relative
            cv2.imwrite(str(output / f"{name}.png"), bgr)
            del pending[name]
    if pending:
        raise RuntimeError(f"could not extract samples: {sorted(pending)}")
    return frames, actual_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", nargs="?", default="bag_extract/20260806_130814")
    parser.add_argument("--output", default="traffic_light_detector/debug_samples")
    parser.add_argument("--analysis", default="traffic_light_detector/analysis/pixel_stats.json")
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    frames, times = extract(Path(args.bag), output)
    report = {"bag": str(Path(args.bag).resolve()), "sample_times_sec": times, "regions": {}}
    for name, (sample, kind, geometry) in REGIONS.items():
        report["regions"][name] = statistics(frames[sample], region_mask(frames[sample].shape,
                                                                          kind, geometry))
    analysis = Path(args.analysis); analysis.parent.mkdir(parents=True, exist_ok=True)
    analysis.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
