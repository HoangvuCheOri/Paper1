#!/usr/bin/env python3
"""ROS 2 CSV logger for paper-ready BSMC trajectory experiments."""

import csv
import math
import os
from datetime import datetime

import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def _yaw_from_quaternion(q):
    t3 = 2.0 * (q.w * q.z + q.x * q.y)
    t4 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)


def _stamp_to_sec(msg):
    if not hasattr(msg, "header"):
        return float("nan")
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class PaperDataLogger(Node):
    """Sample experiment topics into one CSV row at a fixed rate."""

    def __init__(self):
        super().__init__("paper_data_logger")

        self.declare_parameter("output_dir", "~/Paper1/paper_logs")
        self.declare_parameter("output_file", "")
        self.declare_parameter("sample_rate", 25.0)
        self.declare_parameter("controller", "unknown")
        self.declare_parameter("trajectory", "unknown")
        self.declare_parameter("run_id", "")
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("camera_topic", "/odom_camera")
        self.declare_parameter("desired_topic", "/desired_trajectory")
        self.declare_parameter("error_topic", "/tracking_error")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("robot_state_topic", "/robot_state")
        self.declare_parameter("espnow_link_topic", "/espnow_link")

        self.controller = str(self.get_parameter("controller").value)
        self.trajectory = str(self.get_parameter("trajectory").value)
        self.run_id = str(self.get_parameter("run_id").value)

        output_dir = os.path.expanduser(str(self.get_parameter("output_dir").value))
        os.makedirs(output_dir, exist_ok=True)
        output_file = str(self.get_parameter("output_file").value)
        if not output_file:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_controller = self.controller.replace(" ", "_")
            safe_trajectory = self.trajectory.replace(" ", "_")
            safe_run = self.run_id.replace(" ", "_") or "run"
            output_file = f"{stamp}_{safe_trajectory}_{safe_controller}_{safe_run}.csv"
        self.file_path = os.path.join(output_dir, output_file)

        self.odom = {}
        self.camera = {}
        self.desired = {}
        self.error = {}
        self.cmd = {}
        self.robot_state = {}
        self.espnow_link = {}
        self.start_time = None
        self.row_count = 0

        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._odom_cb,
            20,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("camera_topic").value),
            self._camera_cb,
            20,
        )
        self.create_subscription(
            Point,
            str(self.get_parameter("desired_topic").value),
            self._desired_cb,
            20,
        )
        self.create_subscription(
            Point,
            str(self.get_parameter("error_topic").value),
            self._error_cb,
            20,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("cmd_topic").value),
            self._cmd_cb,
            20,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("robot_state_topic").value),
            self._robot_state_cb,
            50,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("espnow_link_topic").value),
            self._espnow_link_cb,
            50,
        )

        self.csv_file = open(self.file_path, "w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self._columns())
        self.writer.writeheader()

        sample_rate = max(1.0, float(self.get_parameter("sample_rate").value))
        self.timer = self.create_timer(1.0 / sample_rate, self._timer_cb)

        self.get_logger().info(
            "Paper data logger started: "
            f"controller={self.controller}, trajectory={self.trajectory}, "
            f"run_id={self.run_id}, file={self.file_path}"
        )

    @staticmethod
    def _columns():
        return [
            "t",
            "ros_time",
            "controller",
            "trajectory",
            "run_id",
            "odom_stamp",
            "odom_x",
            "odom_y",
            "odom_yaw",
            "odom_v",
            "odom_w",
            "camera_stamp",
            "camera_x",
            "camera_y",
            "camera_yaw",
            "desired_stamp",
            "desired_x",
            "desired_y",
            "desired_yaw",
            "error_stamp",
            "error_ex",
            "error_ey",
            "error_etheta",
            "cmd_stamp",
            "cmd_v",
            "cmd_w",
            "robot_state_stamp",
            "rpm_l_x10",
            "rpm_r_x10",
            "gyro_z_x1000",
            "espnow_link_stamp",
            "espnow_rx_time",
            "espnow_robot_time_ms",
            "espnow_seq",
            "espnow_interarrival_ms",
            "espnow_seq_gap",
        ]

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _odom_to_dict(self, msg):
        return {
            "stamp": _stamp_to_sec(msg),
            "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y),
            "yaw": _yaw_from_quaternion(msg.pose.pose.orientation),
            "v": float(msg.twist.twist.linear.x),
            "w": float(msg.twist.twist.angular.z),
        }

    def _odom_cb(self, msg):
        self.odom = self._odom_to_dict(msg)

    def _camera_cb(self, msg):
        self.camera = self._odom_to_dict(msg)

    def _desired_cb(self, msg):
        self.desired = {
            "stamp": self._now_sec(),
            "x": float(msg.x),
            "y": float(msg.y),
            "yaw": float(msg.z),
        }

    def _error_cb(self, msg):
        self.error = {
            "stamp": self._now_sec(),
            "ex": float(msg.x),
            "ey": float(msg.y),
            "etheta": float(msg.z),
        }

    def _cmd_cb(self, msg):
        self.cmd = {
            "stamp": self._now_sec(),
            "v": float(msg.linear.x),
            "w": float(msg.angular.z),
        }

    def _robot_state_cb(self, msg):
        data = list(msg.data)
        self.robot_state = {
            "stamp": self._now_sec(),
            "rpm_l_x10": float(data[0]) if len(data) > 0 else float("nan"),
            "rpm_r_x10": float(data[1]) if len(data) > 1 else float("nan"),
            "gyro_z_x1000": float(data[2]) if len(data) > 2 else float("nan"),
        }

    def _espnow_link_cb(self, msg):
        data = list(msg.data)
        self.espnow_link = {
            "stamp": self._now_sec(),
            "rx_time": float(data[0]) if len(data) > 0 else float("nan"),
            "robot_time_ms": float(data[1]) if len(data) > 1 else float("nan"),
            "seq": float(data[2]) if len(data) > 2 else float("nan"),
            "interarrival_ms": float(data[3]) if len(data) > 3 else float("nan"),
            "seq_gap": float(data[4]) if len(data) > 4 else float("nan"),
        }

    @staticmethod
    def _get(data, key):
        return data.get(key, float("nan"))

    def _timer_cb(self):
        now = self._now_sec()
        if self.start_time is None:
            self.start_time = now

        row = {
            "t": now - self.start_time,
            "ros_time": now,
            "controller": self.controller,
            "trajectory": self.trajectory,
            "run_id": self.run_id,
            "odom_stamp": self._get(self.odom, "stamp"),
            "odom_x": self._get(self.odom, "x"),
            "odom_y": self._get(self.odom, "y"),
            "odom_yaw": self._get(self.odom, "yaw"),
            "odom_v": self._get(self.odom, "v"),
            "odom_w": self._get(self.odom, "w"),
            "camera_stamp": self._get(self.camera, "stamp"),
            "camera_x": self._get(self.camera, "x"),
            "camera_y": self._get(self.camera, "y"),
            "camera_yaw": self._get(self.camera, "yaw"),
            "desired_stamp": self._get(self.desired, "stamp"),
            "desired_x": self._get(self.desired, "x"),
            "desired_y": self._get(self.desired, "y"),
            "desired_yaw": self._get(self.desired, "yaw"),
            "error_stamp": self._get(self.error, "stamp"),
            "error_ex": self._get(self.error, "ex"),
            "error_ey": self._get(self.error, "ey"),
            "error_etheta": self._get(self.error, "etheta"),
            "cmd_stamp": self._get(self.cmd, "stamp"),
            "cmd_v": self._get(self.cmd, "v"),
            "cmd_w": self._get(self.cmd, "w"),
            "robot_state_stamp": self._get(self.robot_state, "stamp"),
            "rpm_l_x10": self._get(self.robot_state, "rpm_l_x10"),
            "rpm_r_x10": self._get(self.robot_state, "rpm_r_x10"),
            "gyro_z_x1000": self._get(self.robot_state, "gyro_z_x1000"),
            "espnow_link_stamp": self._get(self.espnow_link, "stamp"),
            "espnow_rx_time": self._get(self.espnow_link, "rx_time"),
            "espnow_robot_time_ms": self._get(self.espnow_link, "robot_time_ms"),
            "espnow_seq": self._get(self.espnow_link, "seq"),
            "espnow_interarrival_ms": self._get(
                self.espnow_link, "interarrival_ms"
            ),
            "espnow_seq_gap": self._get(self.espnow_link, "seq_gap"),
        }

        self.writer.writerow(row)
        self.row_count += 1
        if self.row_count % 10 == 0:
            self.csv_file.flush()

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PaperDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
