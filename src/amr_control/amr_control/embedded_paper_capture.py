#!/usr/bin/env python3
"""In-process CSV capture and publication export for hardware experiments.

This is deliberately not a ROS node.  A controller owns one instance, so data
logging and final figure export do not increase the number of running nodes.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


def _yaw(msg):
    q = msg.pose.pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _stamp(msg):
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _finite(value):
    return math.isfinite(float(value))


def _safe_name(value):
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))
    return text.strip("_") or "run"


class EmbeddedPaperCapture:
    """Attach subscriptions/timers to an existing controller node."""

    COLUMNS = [
        "t", "ros_time", "controller", "trajectory", "run_id",
        "odom_stamp", "odom_x", "odom_y", "odom_yaw", "odom_v", "odom_w",
        "camera_stamp", "camera_x", "camera_y", "camera_yaw", "camera_age_s",
        "desired_stamp", "desired_x", "desired_y", "desired_yaw",
        "error_stamp", "error_ex", "error_ey", "error_etheta",
        "camera_error_ex", "camera_error_ey", "camera_error_etheta",
        "cmd_stamp", "cmd_v", "cmd_w",
        "espnow_link_stamp", "espnow_rx_time", "espnow_robot_time_ms",
        "espnow_seq", "espnow_interarrival_ms", "espnow_seq_gap",
    ]

    def __init__(
        self,
        node,
        controller,
        trajectory,
        default_duration,
        default_output_dir="~/Paper1/paper_runs",
    ):
        self.node = node
        self.controller = str(controller)
        self.trajectory = str(trajectory)
        self.closed = False
        self.stop_requested = False

        node.declare_parameter("paper_capture", True)
        node.declare_parameter("paper_output_dir", default_output_dir)
        node.declare_parameter("paper_run_id", "")
        node.declare_parameter("paper_sample_rate", 25.0)
        node.declare_parameter("paper_duration", float(default_duration))
        node.declare_parameter("paper_camera_max_age", 0.30)

        self.enabled = bool(node.get_parameter("paper_capture").value)
        if not self.enabled:
            node.get_logger().info("Embedded paper capture disabled.")
            return

        self.duration = float(node.get_parameter("paper_duration").value)
        self.camera_max_age = max(
            0.01, float(node.get_parameter("paper_camera_max_age").value)
        )
        output_dir = Path(os.path.expanduser(
            str(node.get_parameter("paper_output_dir").value)
        ))
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        requested_id = str(node.get_parameter("paper_run_id").value).strip()
        self.run_id = requested_id or stamp
        stem = "_".join((
            stamp,
            _safe_name(self.trajectory),
            _safe_name(self.controller),
            _safe_name(self.run_id),
        ))
        self.output_dir = output_dir
        self.stem = stem
        self.csv_path = output_dir / f"{stem}.csv"
        self.rows = []
        self.latest = {
            "odom": {}, "camera": {}, "desired": {}, "error": {},
            "cmd": {}, "link": {},
        }
        self.start_time = self._now()

        self.stream = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.COLUMNS)
        self.writer.writeheader()

        node.create_subscription(Odometry, "/odometry/filtered", self._odom_cb, 20)
        node.create_subscription(Odometry, "/odom_camera", self._camera_cb, 20)
        node.create_subscription(Point, "/desired_trajectory", self._desired_cb, 20)
        node.create_subscription(Point, "/tracking_error", self._error_cb, 20)
        node.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 20)
        node.create_subscription(Float32MultiArray, "/espnow_link", self._link_cb, 50)

        rate = max(1.0, float(node.get_parameter("paper_sample_rate").value))
        self.sample_timer = node.create_timer(1.0 / rate, self._sample)
        if self.duration > 0.0:
            self.stop_timer = node.create_timer(0.10, self._check_stop)

        duration_text = "manual Ctrl-C" if self.duration == 0.0 else f"{self.duration:.1f}s"
        node.get_logger().info(
            f"Paper capture is embedded in this node: run={self.run_id}, "
            f"duration={duration_text}, csv={self.csv_path}"
        )

    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _odom_cb(self, msg):
        self.latest["odom"] = {
            "stamp": _stamp(msg), "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y), "yaw": _yaw(msg),
            "v": float(msg.twist.twist.linear.x),
            "w": float(msg.twist.twist.angular.z),
        }

    def _camera_cb(self, msg):
        self.latest["camera"] = {
            "stamp": _stamp(msg), "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y), "yaw": _yaw(msg),
        }

    def _desired_cb(self, msg):
        self.latest["desired"] = {
            "stamp": self._now(), "x": float(msg.x),
            "y": float(msg.y), "yaw": float(msg.z),
        }

    def _error_cb(self, msg):
        self.latest["error"] = {
            "stamp": self._now(), "ex": float(msg.x),
            "ey": float(msg.y), "etheta": float(msg.z),
        }

    def _cmd_cb(self, msg):
        self.latest["cmd"] = {
            "stamp": self._now(), "v": float(msg.linear.x),
            "w": float(msg.angular.z),
        }

    def _link_cb(self, msg):
        values = list(msg.data)
        field = lambda index: float(values[index]) if len(values) > index else math.nan
        self.latest["link"] = {
            "stamp": self._now(), "rx_time": field(0), "robot_time_ms": field(1),
            "seq": field(2), "interarrival_ms": field(3), "seq_gap": field(4),
        }

    @staticmethod
    def _get(values, key):
        return values.get(key, math.nan)

    def _sample(self):
        if self.closed:
            return
        now = self._now()
        odom = self.latest["odom"]
        camera = self.latest["camera"]
        desired = self.latest["desired"]
        error = self.latest["error"]
        cmd = self.latest["cmd"]
        link = self.latest["link"]
        camera_stamp = self._get(camera, "stamp")
        camera_age = now - camera_stamp if _finite(camera_stamp) else math.nan

        camera_ex = camera_ey = camera_eth = math.nan
        required = (
            self._get(camera, "x"), self._get(camera, "y"),
            self._get(camera, "yaw"), self._get(desired, "x"),
            self._get(desired, "y"), self._get(desired, "yaw"),
        )
        if all(_finite(value) for value in required) and camera_age <= self.camera_max_age:
            dx = desired["x"] - camera["x"]
            dy = desired["y"] - camera["y"]
            camera_ex = math.cos(camera["yaw"]) * dx + math.sin(camera["yaw"]) * dy
            camera_ey = -math.sin(camera["yaw"]) * dx + math.cos(camera["yaw"]) * dy
            camera_eth = math.atan2(
                math.sin(desired["yaw"] - camera["yaw"]),
                math.cos(desired["yaw"] - camera["yaw"]),
            )

        row = {
            "t": now - self.start_time, "ros_time": now,
            "controller": self.controller, "trajectory": self.trajectory,
            "run_id": self.run_id,
            "odom_stamp": self._get(odom, "stamp"), "odom_x": self._get(odom, "x"),
            "odom_y": self._get(odom, "y"), "odom_yaw": self._get(odom, "yaw"),
            "odom_v": self._get(odom, "v"), "odom_w": self._get(odom, "w"),
            "camera_stamp": camera_stamp, "camera_x": self._get(camera, "x"),
            "camera_y": self._get(camera, "y"), "camera_yaw": self._get(camera, "yaw"),
            "camera_age_s": camera_age,
            "desired_stamp": self._get(desired, "stamp"),
            "desired_x": self._get(desired, "x"), "desired_y": self._get(desired, "y"),
            "desired_yaw": self._get(desired, "yaw"),
            "error_stamp": self._get(error, "stamp"),
            "error_ex": self._get(error, "ex"), "error_ey": self._get(error, "ey"),
            "error_etheta": self._get(error, "etheta"),
            "camera_error_ex": camera_ex, "camera_error_ey": camera_ey,
            "camera_error_etheta": camera_eth,
            "cmd_stamp": self._get(cmd, "stamp"), "cmd_v": self._get(cmd, "v"),
            "cmd_w": self._get(cmd, "w"),
            "espnow_link_stamp": self._get(link, "stamp"),
            "espnow_rx_time": self._get(link, "rx_time"),
            "espnow_robot_time_ms": self._get(link, "robot_time_ms"),
            "espnow_seq": self._get(link, "seq"),
            "espnow_interarrival_ms": self._get(link, "interarrival_ms"),
            "espnow_seq_gap": self._get(link, "seq_gap"),
        }
        self.rows.append(row)
        self.writer.writerow(row)
        if len(self.rows) % 10 == 0:
            self.stream.flush()

    def _check_stop(self):
        if self.stop_requested or self.duration <= 0.0:
            return
        if self._now() - self.start_time < self.duration:
            return
        self.stop_requested = True
        self.node.get_logger().info(
            f"Paper duration {self.duration:.1f}s complete; stopping robot and exporting assets."
        )
        self.node.cmd_pub.publish(Twist())
        # Raising from the executor callback unwinds rclpy.spin immediately;
        # shutting the context inside a callback can leave Humble blocked.
        raise KeyboardInterrupt

    def finalize(self):
        if not self.enabled or self.closed:
            return
        self.closed = True
        self.stream.flush()
        self.stream.close()
        try:
            outputs = self._export_assets()
            from amr_control.publication_aggregator import refresh_tracking_assets
            outputs += refresh_tracking_assets(self.output_dir)
            self.node.get_logger().info(
                "Paper run saved: " + ", ".join(str(path) for path in outputs)
            )
        except Exception as exc:
            self.node.get_logger().error(
                f"CSV is safe at {self.csv_path}, but figure export failed: {exc}"
            )

    def _export_assets(self):
        if len(self.rows) < 3:
            return [self.csv_path]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
            "xtick.labelsize": 7, "ytick.labelsize": 7,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        })
        array = lambda key: np.asarray([float(row[key]) for row in self.rows], dtype=float)
        t = array("t")
        desired_x, desired_y = array("desired_x"), array("desired_y")
        camera_x, camera_y = array("camera_x"), array("camera_y")
        odom_x, odom_y = array("odom_x"), array("odom_y")

        outputs = [self.csv_path]
        fig, ax = plt.subplots(figsize=(3.5, 3.15), constrained_layout=True)
        ax.plot(desired_x, desired_y, "--", color="#222222", lw=1.5, label="Reference")
        ax.plot(odom_x, odom_y, color="#E69F00", lw=1.15, alpha=0.9, label="EKF")
        valid_camera = np.isfinite(camera_x) & np.isfinite(camera_y)
        ax.scatter(camera_x[valid_camera], camera_y[valid_camera], s=4, color="#0072B2",
                   alpha=0.28, edgecolors="none", label="AprilTag")
        valid_desired = np.flatnonzero(np.isfinite(desired_x) & np.isfinite(desired_y))
        if valid_desired.size:
            first = valid_desired[0]
            ax.plot(desired_x[first], desired_y[first], marker="^", ms=6,
                    color="#009E73", linestyle="none", label="Start")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.legend(loc="best", frameon=True)
        outputs += self._save(fig, "trajectory")
        plt.close(fig)

        fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.8), sharex=True, constrained_layout=True)
        error_keys = ("camera_error_ex", "camera_error_ey", "camera_error_etheta")
        labels = (r"$e_x$ (m)", r"$e_y$ (m)", r"$e_\theta$ (rad)")
        colors = ("#0072B2", "#E69F00", "#D55E00")
        for ax, key, label, color in zip(axes, error_keys, labels, colors):
            ax.plot(t, array(key), color=color, lw=0.9)
            ax.axhline(0.0, color="#555555", lw=0.5, ls="--")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3, lw=0.5)
        axes[-1].set_xlabel("time (s)")
        outputs += self._save(fig, "errors")
        plt.close(fig)

        cmd_v, cmd_w = array("cmd_v"), array("cmd_w")
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.25), sharex=True, constrained_layout=True)
        axes[0].plot(t, cmd_v, color="#0072B2", lw=0.9)
        axes[0].set_ylabel(r"$v_{cmd}$ (m/s)")
        axes[1].plot(t, cmd_w, color="#D55E00", lw=0.9)
        axes[1].set_ylabel(r"$\omega_{cmd}$ (rad/s)")
        axes[1].set_xlabel("time (s)")
        for ax in axes:
            ax.axhline(0.0, color="#555555", lw=0.5, ls="--")
            ax.grid(True, alpha=0.3, lw=0.5)
        outputs += self._save(fig, "commands")
        plt.close(fig)

        moving = np.isfinite(cmd_v) & (np.abs(cmd_v) > 0.005)
        if moving.any():
            first_moving_t = t[np.flatnonzero(moving)[0]]
            metric_mask = moving & (t >= first_moving_t + 5.0)
        else:
            metric_mask = np.ones_like(t, dtype=bool)
        ex, ey, eth = (array(key) for key in error_keys)
        metric_mask &= np.isfinite(ex) & np.isfinite(ey) & np.isfinite(eth)
        summary = {
            "controller": self.controller, "trajectory": self.trajectory,
            "run_id": self.run_id, "csv": str(self.csv_path),
            "n_total": int(len(t)), "n_metric": int(metric_mask.sum()),
            "valid": bool(metric_mask.sum() >= 3),
            "camera_max_age_s": self.camera_max_age,
            "parameters": self._controller_parameters(),
        }
        if metric_mask.any():
            summary.update({
                "camera_rmse_ex_m": float(np.sqrt(np.mean(ex[metric_mask] ** 2))),
                "camera_rmse_ey_m": float(np.sqrt(np.mean(ey[metric_mask] ** 2))),
                "camera_rmse_heading_deg": float(np.degrees(np.sqrt(np.mean(eth[metric_mask] ** 2)))),
                "camera_rmse_position_m": float(np.sqrt(np.mean(
                    ex[metric_mask] ** 2 + ey[metric_mask] ** 2
                ))),
            })
        if self.trajectory == "eight" and metric_mask.any():
            desired_yaw = array("desired_yaw")
            center_x = 0.5 * (np.nanmin(desired_x) + np.nanmax(desired_x))
            center_y = 0.5 * (np.nanmin(desired_y) + np.nanmax(desired_y))
            center = np.hypot(desired_x - center_x, desired_y - center_y) <= 0.15
            dx_camera = desired_x - camera_x
            dy_camera = desired_y - camera_y
            lateral = (
                -np.sin(desired_yaw) * dx_camera
                + np.cos(desired_yaw) * dy_camera
            )
            valid_crossing = metric_mask & center & np.isfinite(lateral)
            directions = {
                "left_to_right": valid_crossing & (np.cos(desired_yaw) > 0.0),
                "right_to_left": valid_crossing & (np.cos(desired_yaw) < 0.0),
            }
            for direction, mask in directions.items():
                if mask.any():
                    summary[f"crossing_lateral_rmse_{direction}_m"] = float(
                        np.sqrt(np.mean(lateral[mask] ** 2))
                    )
                    summary[f"crossing_lateral_bias_{direction}_m"] = float(
                        np.mean(lateral[mask])
                    )
        summary_path = self.output_dir / f"{self.stem}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        outputs.append(summary_path)
        return outputs

    def _controller_parameters(self):
        aliases = {
            "k1": "k1", "k2": "k2", "k3": "k3",
            "k2_straight": "k2_straight", "k3_straight": "k3_straight",
            "Ks1": "ks1", "Ks2": "ks2", "phi1": "phi1", "phi2": "phi2",
            "R": "radius_m", "W": "angular_speed_rad_s", "A": "amplitude_m",
            "side_length": "side_length_m", "VD": "desired_speed_m_s",
            "MAX_V": "max_v_m_s", "MAX_W": "max_w_rad_s",
            "CENTER_K1": "center_k1", "CENTER_K1_RADIUS": "center_k1_radius_m",
            "CENTER_K2": "center_k2", "CENTER_K2_RADIUS": "center_k2_radius_m",
            "CENTER_K3": "center_k3", "CENTER_K3_RADIUS": "center_k3_radius_m",
            "START_PHASE_DEG": "start_phase_deg",
            "PATH_ROTATION_DEG": "path_rotation_deg",
        }
        values = {}
        for attribute, key in aliases.items():
            if hasattr(self.node, attribute):
                value = getattr(self.node, attribute)
                if isinstance(value, (int, float)):
                    values[key] = float(value)
        return values

    def _save(self, fig, suffix):
        paths = []
        for extension, options in (("pdf", {}), ("png", {"dpi": 600})):
            path = self.output_dir / f"{self.stem}_{suffix}.{extension}"
            fig.savefig(path, bbox_inches="tight", **options)
            paths.append(path)
        return paths
