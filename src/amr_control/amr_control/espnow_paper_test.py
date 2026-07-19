#!/usr/bin/env python3
"""One-node ESP-NOW measurement, CSV logger, summary, and figure exporter."""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class EspNowPaperTest(Node):
    COLUMNS = [
        "t", "ros_time", "condition", "distance_m", "run_id",
        "rx_time", "robot_time_ms", "seq", "interarrival_ms", "seq_gap",
    ]

    def __init__(self):
        super().__init__("espnow_paper_test")
        self.declare_parameter("condition", "static")
        self.declare_parameter("distance_m", 5.0)
        self.declare_parameter("duration", 120.0)
        self.declare_parameter("run_id", "")
        self.declare_parameter("output_dir", "~/Paper1/paper_runs/espnow")
        self.declare_parameter("link_topic", "/espnow_link")
        self.declare_parameter("nominal_period_ms", 50.0)

        self.condition = str(self.get_parameter("condition").value).strip()
        self.distance_m = float(self.get_parameter("distance_m").value)
        self.duration = max(0.0, float(self.get_parameter("duration").value))
        self.nominal_ms = max(
            1.0, float(self.get_parameter("nominal_period_ms").value)
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        requested_run = str(self.get_parameter("run_id").value).strip()
        self.run_id = requested_run or stamp
        output_dir = Path(os.path.expanduser(
            str(self.get_parameter("output_dir").value)
        ))
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_condition = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in self.condition
        ) or "unknown"
        self.stem = f"{stamp}_espnow_{safe_condition}_{self.distance_m:g}m_{self.run_id}"
        self.output_dir = output_dir
        self.csv_path = output_dir / f"{self.stem}.csv"
        self.stream = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.COLUMNS)
        self.writer.writeheader()
        self.rows = []
        self.start_time = self._now()
        self.closed = False
        self.stop_requested = False

        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("link_topic").value),
            self._link_cb,
            100,
        )
        if self.duration > 0.0:
            self.create_timer(0.10, self._check_stop)
        duration_text = "manual Ctrl-C" if self.duration == 0.0 else f"{self.duration:.1f}s"
        self.get_logger().info(
            f"ESP-NOW paper test: {self.condition}, {self.distance_m:g}m, "
            f"duration={duration_text}, output={self.csv_path}"
        )

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _field(values, index):
        return float(values[index]) if len(values) > index else math.nan

    def _link_cb(self, msg):
        now = self._now()
        values = list(msg.data)
        row = {
            "t": now - self.start_time,
            "ros_time": now,
            "condition": self.condition,
            "distance_m": self.distance_m,
            "run_id": self.run_id,
            "rx_time": self._field(values, 0),
            "robot_time_ms": self._field(values, 1),
            "seq": self._field(values, 2),
            "interarrival_ms": self._field(values, 3),
            "seq_gap": self._field(values, 4),
        }
        self.rows.append(row)
        self.writer.writerow(row)
        if len(self.rows) % 10 == 0:
            self.stream.flush()

    def _check_stop(self):
        if self.stop_requested or self._now() - self.start_time < self.duration:
            return
        self.stop_requested = True
        self.get_logger().info("ESP-NOW measurement complete; exporting paper assets.")
        # Unwind rclpy.spin; Context.shutdown() from inside a Humble callback
        # can leave the executor waiting indefinitely.
        raise KeyboardInterrupt

    def finalize(self):
        if self.closed:
            return
        self.closed = True
        self.stream.flush()
        self.stream.close()
        try:
            outputs = self._export()
            from amr_control.publication_aggregator import refresh_link_assets
            outputs += refresh_link_assets(self.output_dir, self.nominal_ms)
            self.get_logger().info(
                "ESP-NOW results saved: " + ", ".join(str(path) for path in outputs)
            )
        except Exception as exc:
            self.get_logger().error(
                f"CSV is safe at {self.csv_path}, but export failed: {exc}"
            )

    def _export(self):
        outputs = [self.csv_path]
        if not self.rows:
            summary = {
                "condition": self.condition, "distance_m": self.distance_m,
                "run_id": self.run_id, "packets": 0,
                "valid": False, "reason": "No /espnow_link messages received",
            }
            summary_path = self.output_dir / f"{self.stem}_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return outputs + [summary_path]

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
        interarrival = array("interarrival_ms")
        seq = array("seq")
        seq_gap = array("seq_gap")
        valid_ia = np.isfinite(interarrival) & (interarrival >= 0.0)
        ia = interarrival[valid_ia]
        valid_seq = np.isfinite(seq)
        explicit_gaps = seq_gap[np.isfinite(seq_gap) & (seq_gap >= 0.0)]
        if explicit_gaps.size:
            missing = int(np.rint(explicit_gaps).sum())
            loss_method = "packet sequence numbers"
        else:
            missing = int(np.maximum(0, np.rint(ia / self.nominal_ms).astype(int) - 1).sum())
            loss_method = "inter-arrival gap estimate"
        sent_estimate = len(self.rows) + missing
        loss_percent = 100.0 * missing / max(1, sent_estimate)

        summary = {
            "condition": self.condition, "distance_m": self.distance_m,
            "run_id": self.run_id, "duration_s": float(t[-1]) if len(t) else 0.0,
            "packets_received": int(len(self.rows)), "missing_packets": missing,
            "loss_percent": loss_percent, "loss_method": loss_method,
            "sequence_available": bool(valid_seq.any()),
            "robot_timestamp_available": bool(np.isfinite(array("robot_time_ms")).any()),
            "nominal_period_ms": self.nominal_ms,
        }
        if ia.size:
            summary.update({
                "median_interarrival_ms": float(np.median(ia)),
                "p95_interarrival_ms": float(np.percentile(ia, 95.0)),
                "p99_interarrival_ms": float(np.percentile(ia, 99.0)),
                "jitter_std_ms": float(np.std(ia, ddof=1)) if ia.size > 1 else 0.0,
            })
        summary_path = self.output_dir / f"{self.stem}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        outputs.append(summary_path)

        fig, ax = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
        ax.plot(t[valid_ia], ia, color="#0072B2", lw=0.75, alpha=0.85)
        ax.axhline(self.nominal_ms, color="#222222", ls="--", lw=0.8,
                   label=f"Nominal {self.nominal_ms:g} ms")
        gap_mask = np.isfinite(seq_gap) & (seq_gap > 0.0) & valid_ia
        if gap_mask.any():
            ax.scatter(t[gap_mask], interarrival[gap_mask], marker="x", s=18,
                       color="#D55E00", label="Sequence gap", zorder=3)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("inter-arrival time (ms)")
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.legend(loc="best")
        outputs += self._save(fig, "timeseries")
        plt.close(fig)

        if ia.size:
            fig, ax = plt.subplots(figsize=(2.5, 2.75), constrained_layout=True)
            ax.boxplot(
                [ia], labels=[f"{self.condition}\n{self.distance_m:g} m"],
                showfliers=True, widths=0.45,
                medianprops={"color": "#D55E00", "linewidth": 1.2},
                boxprops={"color": "#0072B2"},
                whiskerprops={"color": "#0072B2"},
                capprops={"color": "#0072B2"},
                flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35},
            )
            ax.axhline(self.nominal_ms, color="#222222", ls="--", lw=0.8)
            ax.text(1.0, ax.get_ylim()[1], f"loss={loss_percent:.2f}%",
                    ha="center", va="top", fontsize=7)
            ax.set_ylabel("inter-arrival time (ms)")
            ax.grid(True, axis="y", alpha=0.3, lw=0.5)
            outputs += self._save(fig, "distribution")
            plt.close(fig)
        return outputs

    def _save(self, fig, suffix):
        outputs = []
        for extension, options in (("pdf", {}), ("png", {"dpi": 600})):
            path = self.output_dir / f"{self.stem}_{suffix}.{extension}"
            fig.savefig(path, bbox_inches="tight", **options)
            outputs.append(path)
        return outputs


def main(args=None):
    rclpy.init(args=args)
    node = EspNowPaperTest()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.finalize()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
