#!/usr/bin/env python3
"""Aggregate detector topics for repeatable rosbag validation."""

import argparse
from collections import Counter
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, String


class Monitor(Node):
    def __init__(self, duration):
        super().__init__("traffic_light_validation_monitor")
        self.deadline = time.monotonic() + duration; self.states = Counter(); self.transitions = []
        self.scores = []; self.debug_count = 0; self.mask_count = 0; self.first = time.monotonic()
        self._validation_subscriptions = [
            self.create_subscription(String, "/traffic_light/state", self.on_state, 50),
            self.create_subscription(Float32MultiArray, "/traffic_light/scores", self.on_scores, 50),
            self.create_subscription(CompressedImage, "/traffic_light/debug/compressed", self.on_debug, 10),
            self.create_subscription(CompressedImage, "/traffic_light/mask/compressed", self.on_mask, 10),
        ]

    def on_state(self, message):
        self.states[message.data] += 1
        if not self.transitions or self.transitions[-1][1] != message.data:
            self.transitions.append((round(time.monotonic() - self.first, 3), message.data))

    def on_scores(self, message):
        if len(message.data) >= 6:
            self.scores.append(list(message.data[:6]))

    def on_debug(self, _message): self.debug_count += 1
    def on_mask(self, _message): self.mask_count += 1

    def report(self):
        elapsed = max(1e-6, time.monotonic() - self.first)
        array = np.asarray(self.scores, dtype=float)
        return {
            "elapsed_sec": round(elapsed, 3), "state_counts": dict(self.states),
            "transitions": self.transitions,
            "state_hz": round(sum(self.states.values()) / elapsed, 3),
            "debug_hz": round(self.debug_count / elapsed, 3),
            "mask_hz": round(self.mask_count / elapsed, 3),
            "score_order": ["red", "yellow", "left", "green", "detection", "white_light"],
            "score_max": np.max(array, axis=0).round(4).tolist() if array.size else [],
            "score_mean": np.mean(array, axis=0).round(4).tolist() if array.size else [],
        }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args(); rclpy.init(); node = Monitor(args.duration)
    while rclpy.ok() and time.monotonic() < node.deadline:
        rclpy.spin_once(node, timeout_sec=.1)
    print(json.dumps(node.report(), indent=2)); node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__": main()
