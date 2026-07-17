#!/usr/bin/env python3
"""Packet-level ESP-NOW link logger for paper metrics."""

import csv
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class PaperLinkLogger(Node):
    """Write one CSV row per /espnow_link message."""

    def __init__(self):
        super().__init__("paper_link_logger")

        self.declare_parameter("output_dir", "~/Paper1/paper_logs")
        self.declare_parameter("output_file", "")
        self.declare_parameter("condition", "unknown")
        self.declare_parameter("distance_m", float("nan"))
        self.declare_parameter("duration", 0.0)
        self.declare_parameter("link_topic", "/espnow_link")

        output_dir = os.path.expanduser(str(self.get_parameter("output_dir").value))
        os.makedirs(output_dir, exist_ok=True)
        output_file = str(self.get_parameter("output_file").value)
        if not output_file:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            condition = str(self.get_parameter("condition").value).replace(" ", "_")
            output_file = f"{stamp}_espnow_{condition}.csv"
        self.file_path = os.path.join(output_dir, output_file)

        self.condition = str(self.get_parameter("condition").value)
        self.distance_m = float(self.get_parameter("distance_m").value)
        self.duration = float(self.get_parameter("duration").value)
        self.start_time = None
        self.rows = 0

        self.csv_file = open(self.file_path, "w", newline="")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "t",
                "ros_time",
                "condition",
                "distance_m",
                "rx_time",
                "robot_time_ms",
                "seq",
                "interarrival_ms",
                "seq_gap",
            ],
        )
        self.writer.writeheader()

        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("link_topic").value),
            self._link_cb,
            100,
        )

        self.get_logger().info(
            f"Paper link logger started: condition={self.condition}, "
            f"distance_m={self.distance_m}, duration={self.duration}s, file={self.file_path}"
        )

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _field(data, index):
        return float(data[index]) if len(data) > index else float("nan")

    def _link_cb(self, msg):
        now = self._now_sec()
        if self.start_time is None:
            self.start_time = now

        data = list(msg.data)
        row = {
            "t": now - self.start_time,
            "ros_time": now,
            "condition": self.condition,
            "distance_m": self.distance_m,
            "rx_time": self._field(data, 0),
            "robot_time_ms": self._field(data, 1),
            "seq": self._field(data, 2),
            "interarrival_ms": self._field(data, 3),
            "seq_gap": self._field(data, 4),
        }
        self.writer.writerow(row)
        self.rows += 1
        if self.rows % 10 == 0:
            self.csv_file.flush()

        if self.duration > 0.0 and (now - self.start_time) >= self.duration:
            self.get_logger().info(f"Đã đạt giới hạn {self.duration}s. Tự động tắt logger.")
            raise SystemExit

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PaperLinkLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
