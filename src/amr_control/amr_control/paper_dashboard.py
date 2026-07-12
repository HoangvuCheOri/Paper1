#!/usr/bin/env python3
"""Real-time dashboard for BSMC paper experiments.

Panels:
  1. XY trajectory  — actual (EKF filtered) vs desired
  2. v_cmd & w_cmd  — command velocities over time
  3. Tracking errors — e_x, e_y, e_theta over time
  4. ESP-NOW link    — inter-arrival jitter & packet loss indicator
"""

import math
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

import matplotlib
matplotlib.use("TkAgg")  # noqa: E402  — must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------
class PaperDashboardNode(Node):
    """Collects data from relevant topics for the dashboard."""

    MAXLEN_XY = 60_000   # ~40 min @ 25 Hz
    MAXLEN_TS = 15_000   # ~10 min @ 25 Hz

    def __init__(self):
        super().__init__("paper_dashboard")

        # ---- parameters ----
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("desired_topic", "/desired_trajectory")
        self.declare_parameter("error_topic", "/tracking_error")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("espnow_link_topic", "/espnow_link")
        self.declare_parameter("camera_topic", "/odom_camera")

        odom_topic = str(self.get_parameter("odom_topic").value)
        desired_topic = str(self.get_parameter("desired_topic").value)
        error_topic = str(self.get_parameter("error_topic").value)
        cmd_topic = str(self.get_parameter("cmd_topic").value)
        espnow_topic = str(self.get_parameter("espnow_link_topic").value)
        camera_topic = str(self.get_parameter("camera_topic").value)

        # ---- XY history ----
        self.filtered_x = deque(maxlen=self.MAXLEN_XY)
        self.filtered_y = deque(maxlen=self.MAXLEN_XY)
        self.desired_x = deque(maxlen=self.MAXLEN_XY)
        self.desired_y = deque(maxlen=self.MAXLEN_XY)
        self.camera_x = deque(maxlen=self.MAXLEN_XY)
        self.camera_y = deque(maxlen=self.MAXLEN_XY)

        # ---- time-series history ----
        self.cmd_t = deque(maxlen=self.MAXLEN_TS)
        self.cmd_v = deque(maxlen=self.MAXLEN_TS)
        self.cmd_w = deque(maxlen=self.MAXLEN_TS)

        self.err_t = deque(maxlen=self.MAXLEN_TS)
        self.err_ex = deque(maxlen=self.MAXLEN_TS)
        self.err_ey = deque(maxlen=self.MAXLEN_TS)
        self.err_eth = deque(maxlen=self.MAXLEN_TS)

        self.link_t = deque(maxlen=self.MAXLEN_TS)
        self.link_interarrival = deque(maxlen=self.MAXLEN_TS)
        self.link_seq_gap = deque(maxlen=self.MAXLEN_TS)

        self.start_time = None

        # ---- subscriptions ----
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 20)
        self.create_subscription(Point, desired_topic, self._desired_cb, 20)
        self.create_subscription(Point, error_topic, self._error_cb, 20)
        self.create_subscription(Twist, cmd_topic, self._cmd_cb, 20)
        self.create_subscription(Float32MultiArray, espnow_topic, self._link_cb, 50)
        self.create_subscription(Odometry, camera_topic, self._camera_cb, 20)

        self.get_logger().info(
            f"Paper dashboard started — odom={odom_topic}, "
            f"desired={desired_topic}, error={error_topic}, "
            f"cmd={cmd_topic}, link={espnow_topic}, camera={camera_topic}"
        )

    # ---- helpers ----
    def _now(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.start_time is None:
            self.start_time = t
        return t - self.start_time

    @staticmethod
    def _yaw(msg):
        q = msg.pose.pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    # ---- callbacks ----
    def _odom_cb(self, msg):
        self.filtered_x.append(msg.pose.pose.position.x)
        self.filtered_y.append(msg.pose.pose.position.y)

    def _camera_cb(self, msg):
        self.camera_x.append(msg.pose.pose.position.x)
        self.camera_y.append(msg.pose.pose.position.y)

    def _desired_cb(self, msg):
        self.desired_x.append(msg.x)
        self.desired_y.append(msg.y)

    def _error_cb(self, msg):
        t = self._now()
        self.err_t.append(t)
        self.err_ex.append(msg.x)
        self.err_ey.append(msg.y)
        self.err_eth.append(msg.z)

    def _cmd_cb(self, msg):
        t = self._now()
        self.cmd_t.append(t)
        self.cmd_v.append(msg.linear.x)
        self.cmd_w.append(msg.angular.z)

    def _link_cb(self, msg):
        t = self._now()
        data = list(msg.data)
        self.link_t.append(t)
        self.link_interarrival.append(float(data[3]) if len(data) > 3 else float("nan"))
        self.link_seq_gap.append(float(data[4]) if len(data) > 4 else float("nan"))


# ---------------------------------------------------------------------------
# Matplotlib dashboard
# ---------------------------------------------------------------------------
_COLORS = {
    "bg": "#ffffff",
    "panel": "#ffffff",
    "grid": "#cccccc",
    "text": "#222222",
    "desired": "#d32f2f",
    "actual": "#1565c0",
    "camera": "#2e7d32",
    "vcmd": "#e65100",
    "wcmd": "#6a1b9a",
    "ex": "#00838f",
    "ey": "#e65100",
    "eth": "#c62828",
    "link_ok": "#2e7d32",
    "link_bad": "#c62828",
    "jitter": "#7b1fa2",
}


def _style_ax(ax, title):
    """Apply dark-theme styling to an axes."""
    ax.set_facecolor(_COLORS["panel"])
    ax.set_title(title, color=_COLORS["text"], fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=_COLORS["text"], labelsize=8)
    ax.xaxis.label.set_color(_COLORS["text"])
    ax.yaxis.label.set_color(_COLORS["text"])
    ax.grid(True, color=_COLORS["grid"], alpha=0.35, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(_COLORS["grid"])


def run_dashboard(node: PaperDashboardNode):
    """Build and run the Matplotlib dashboard."""

    fig = plt.figure(figsize=(14, 9), facecolor=_COLORS["bg"])
    fig.canvas.manager.set_window_title("BSMC Paper Dashboard")

    # ----- layout: 2×2 grid, bottom row spans 2 cols for link -----
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30,
                          left=0.06, right=0.97, top=0.95, bottom=0.06)

    # Panel 1 — XY trajectory
    ax_xy = fig.add_subplot(gs[0, 0])
    _style_ax(ax_xy, "Trajectory (X – Y)")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ln_desired, = ax_xy.plot([], [], "--", color=_COLORS["desired"], lw=2, label="Desired")
    ln_actual, = ax_xy.plot([], [], "-", color=_COLORS["actual"], lw=2, label="EKF Filtered")
    ln_camera, = ax_xy.plot([], [], ".", color=_COLORS["camera"], ms=2.5, alpha=0.45, label="Camera")
    ax_xy.legend(loc="upper left", fontsize=7, facecolor=_COLORS["panel"],
                 edgecolor=_COLORS["grid"], labelcolor=_COLORS["text"])

    # Panel 2 — Command velocities
    ax_cmd = fig.add_subplot(gs[0, 1])
    _style_ax(ax_cmd, "Command Velocities")
    ax_cmd.set_xlabel("time (s)")
    ax_cmd.set_ylabel("value")
    ln_vcmd, = ax_cmd.plot([], [], "-", color=_COLORS["vcmd"], lw=1.5, label="v_cmd (m/s)")
    ln_wcmd, = ax_cmd.plot([], [], "-", color=_COLORS["wcmd"], lw=1.5, label="ω_cmd (rad/s)")
    ax_cmd.axhline(0, color=_COLORS["text"], ls="--", alpha=0.25)
    ax_cmd.legend(loc="upper right", fontsize=7, facecolor=_COLORS["panel"],
                  edgecolor=_COLORS["grid"], labelcolor=_COLORS["text"])

    # Panel 3 — Tracking errors
    ax_err = fig.add_subplot(gs[1, :])
    _style_ax(ax_err, "Tracking Errors")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("error")
    ln_ex, = ax_err.plot([], [], "-", color=_COLORS["ex"], lw=1.4, label="eₓ (m)")
    ln_ey, = ax_err.plot([], [], "-", color=_COLORS["ey"], lw=1.4, label="eᵧ (m)")
    ln_eth, = ax_err.plot([], [], "-", color=_COLORS["eth"], lw=1.4, label="eθ (rad)")
    ax_err.axhline(0, color=_COLORS["text"], ls="--", alpha=0.25)
    ax_err.legend(loc="upper right", fontsize=7, ncol=3, facecolor=_COLORS["panel"],
                  edgecolor=_COLORS["grid"], labelcolor=_COLORS["text"])

    # Panel 4 — ESP-NOW link quality
    ax_link = fig.add_subplot(gs[2, :])
    _style_ax(ax_link, "ESP-NOW Link Quality")
    ax_link.set_xlabel("time (s)")
    ax_link.set_ylabel("inter-arrival (ms)")
    ln_jitter, = ax_link.plot([], [], "-", color=_COLORS["jitter"], lw=1.2, alpha=0.8,
                              label="Inter-arrival (ms)")
    # Scatter overlay for seq gaps (dropped packets)
    sc_gap = ax_link.scatter([], [], c=_COLORS["link_bad"], s=18, marker="x",
                             zorder=5, label="Seq gap (loss)")
    ax_link.axhline(50, color=_COLORS["link_ok"], ls=":", alpha=0.5, label="Nominal 50 ms")
    ax_link.legend(loc="upper right", fontsize=7, ncol=3, facecolor=_COLORS["panel"],
                   edgecolor=_COLORS["grid"], labelcolor=_COLORS["text"])

    # ---- status text (top center) ----
    status_text = fig.text(
        0.50, 0.98, "", ha="center", va="top",
        fontsize=9, color=_COLORS["text"],
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=_COLORS["panel"],
                  edgecolor=_COLORS["grid"], alpha=0.9),
    )

    # ------------------------------------------------------------------
    def _update(_frame):
        artists = []

        # ---- XY trajectory ----
        if len(node.desired_x) > 1 or len(node.filtered_x) > 1:
            ln_desired.set_data(list(node.desired_x), list(node.desired_y))
            ln_actual.set_data(list(node.filtered_x), list(node.filtered_y))
            ln_camera.set_data(list(node.camera_x), list(node.camera_y))
            ax_xy.relim()
            ax_xy.autoscale_view()
            artists += [ln_desired, ln_actual, ln_camera]

        # ---- Command velocities ----
        if len(node.cmd_t) > 1:
            t_list = list(node.cmd_t)
            ln_vcmd.set_data(t_list, list(node.cmd_v))
            ln_wcmd.set_data(t_list, list(node.cmd_w))
            ax_cmd.set_xlim(0, max(10, t_list[-1] + 1))
            all_vals = list(node.cmd_v) + list(node.cmd_w)
            if all_vals:
                lo = min(all_vals)
                hi = max(all_vals)
                margin = max(0.05, (hi - lo) * 0.15)
                ax_cmd.set_ylim(lo - margin, hi + margin)
            artists += [ln_vcmd, ln_wcmd]

        # ---- Tracking errors ----
        if len(node.err_t) > 1:
            t_list = list(node.err_t)
            ln_ex.set_data(t_list, list(node.err_ex))
            ln_ey.set_data(t_list, list(node.err_ey))
            ln_eth.set_data(t_list, list(node.err_eth))
            ax_err.set_xlim(0, max(10, t_list[-1] + 1))
            all_errs = list(node.err_ex) + list(node.err_ey) + list(node.err_eth)
            if all_errs:
                bound = max(abs(min(all_errs)), abs(max(all_errs)), 0.05) * 1.15
                ax_err.set_ylim(-bound, bound)
            artists += [ln_ex, ln_ey, ln_eth]

        # ---- ESP-NOW link ----
        if len(node.link_t) > 1:
            t_arr = np.array(node.link_t)
            ia_arr = np.array(node.link_interarrival)
            gap_arr = np.array(node.link_seq_gap)

            finite_mask = np.isfinite(ia_arr)
            ln_jitter.set_data(t_arr[finite_mask], ia_arr[finite_mask])

            # Mark seq gaps > 1 (= missing packets)
            gap_mask = np.isfinite(gap_arr) & (gap_arr > 1.0)
            if gap_mask.any():
                sc_gap.set_offsets(np.column_stack([t_arr[gap_mask], ia_arr[gap_mask]]))
            else:
                sc_gap.set_offsets(np.empty((0, 2)))

            ax_link.set_xlim(0, max(10, t_arr[-1] + 1))
            if finite_mask.any():
                ia_finite = ia_arr[finite_mask]
                lo = max(0, float(np.nanmin(ia_finite)) - 10)
                hi = float(np.nanmax(ia_finite)) + 15
                ax_link.set_ylim(lo, hi)
            artists += [ln_jitter, sc_gap]

        # ---- status bar ----
        n_err = len(node.err_t)
        n_link = len(node.link_t)
        elapsed = f"{node.err_t[-1]:.1f}s" if n_err else "—"

        # Compute RMSE from last 50 error samples
        if n_err > 5:
            recent_ex = list(node.err_ex)[-50:]
            recent_ey = list(node.err_ey)[-50:]
            rmse_p = math.sqrt(
                sum(x * x + y * y for x, y in zip(recent_ex, recent_ey))
                / len(recent_ex)
            )
            err_str = f"RMSE_p(50)={rmse_p:.4f}m"
        else:
            err_str = "waiting…"

        # Link loss rate
        if n_link > 10:
            gaps = np.array(node.link_seq_gap)
            finite_gaps = gaps[np.isfinite(gaps)]
            if finite_gaps.size > 0:
                lost = int(np.sum(finite_gaps[finite_gaps > 1] - 1))
                total = len(finite_gaps) + lost
                loss_pct = 100.0 * lost / total if total > 0 else 0.0
                link_str = f"Loss={loss_pct:.1f}%"
            else:
                link_str = "no seq"
        else:
            link_str = "no link data"

        status_text.set_text(
            f"  t={elapsed}  |  samples: err={n_err} link={n_link}  "
            f"|  {err_str}  |  ESP-NOW {link_str}  "
        )
        artists.append(status_text)

        return artists

    _ani = animation.FuncAnimation(  # noqa: F841
        fig, _update, interval=120, blit=False, cache_frame_data=False,
    )
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = PaperDashboardNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_dashboard(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
