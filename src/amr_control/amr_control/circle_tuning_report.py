#!/usr/bin/env python3
"""Create a tuning report focused on circular path quality and transients."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _number(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def _rmse(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else math.nan


def _wrap(values):
    return np.arctan2(np.sin(values), np.cos(values))


def _smooth(values, t, window_s=2.0):
    dt = float(np.median(np.diff(t)))
    width = max(3, int(round(window_s / max(dt, 1e-3))))
    if width % 2 == 0:
        width += 1
    width = min(width, len(values) - (1 - len(values) % 2))
    if width < 3:
        return np.full_like(values, np.mean(values))
    pad = width // 2
    padded = np.pad(values, pad, mode="reflect")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _fit_circle(x, y):
    """Least-squares circle fitted to the sampled desired reference."""
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(x))))
    target = x * x + y * y
    center_x, center_y, constant = np.linalg.lstsq(
        matrix, target, rcond=None
    )[0]
    radius_sq = constant + center_x * center_x + center_y * center_y
    return float(center_x), float(center_y), math.sqrt(max(0.0, radius_sq))


def analyze(input_path, output_dir=None, transient_s=5.0):
    input_path = Path(input_path).expanduser().resolve()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir else input_path.parent / "circle_reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = [
        "t", "ros_time", "camera_x", "camera_y", "camera_yaw",
        "desired_stamp", "desired_x", "desired_y", "desired_yaw",
        "cmd_stamp", "cmd_v", "cmd_w",
    ]
    data = {key: np.asarray([_number(row, key) for row in rows]) for key in keys}
    valid = np.ones(len(rows), dtype=bool)
    for key in keys:
        valid &= np.isfinite(data[key])
    valid &= np.abs(data["ros_time"] - data["desired_stamp"]) <= 0.30
    valid &= np.abs(data["ros_time"] - data["cmd_stamp"]) <= 0.30
    if int(valid.sum()) < 100:
        raise ValueError("CSV does not contain enough fresh circle samples")
    for key in keys:
        data[key] = data[key][valid]

    moving = np.flatnonzero(data["cmd_v"] > 0.01)
    if len(moving) == 0:
        raise ValueError("CSV does not contain forward-motion samples")
    motion_start = int(moving[0])
    fresh_t = data["t"] - data["t"][0]
    pre_motion_duration = float(fresh_t[motion_start])
    pre_motion_cmd_w_peak = float(
        np.max(np.abs(data["cmd_w"][:motion_start + 1]))
    )
    for key in keys:
        data[key] = data[key][motion_start:]
    t = data["t"] - data["t"][0]

    center_x, center_y, reference_radius = _fit_circle(
        data["desired_x"], data["desired_y"]
    )
    dx = data["desired_x"] - data["camera_x"]
    dy = data["desired_y"] - data["camera_y"]
    position = np.hypot(dx, dy)
    radial_error = (
        np.hypot(data["camera_x"] - center_x, data["camera_y"] - center_y)
        - reference_radius
    )
    heading = _wrap(data["desired_yaw"] - data["camera_yaw"])
    radial_trend = _smooth(radial_error, t)
    ripple = radial_error - radial_trend
    steady = t >= transient_s
    if int(steady.sum()) < 50:
        raise ValueError("CSV is too short after the transient interval")
    dt = np.diff(t)
    dw_dt = np.diff(data["cmd_w"]) / np.maximum(dt, 1e-3)

    summary = {
        "input": str(input_path),
        "active_duration_s": float(t[-1]),
        "pre_motion_duration_s": pre_motion_duration,
        "pre_motion_cmd_w_peak_rad_s": pre_motion_cmd_w_peak,
        "n_samples": int(len(t)),
        "reference_center_x_m": center_x,
        "reference_center_y_m": center_y,
        "reference_radius_m": reference_radius,
        "position_rmse_m": _rmse(position[steady]),
        "position_max_m": float(np.max(position[steady])),
        "path_rmse_m": _rmse(radial_error[steady]),
        "path_max_m": float(np.max(np.abs(radial_error[steady]))),
        "radial_bias_m": float(np.mean(radial_error[steady])),
        "radial_waviness_rmse_m": _rmse(ripple[steady]),
        "radial_waviness_peak_to_peak_m": float(np.ptp(ripple[steady])),
        "heading_rmse_deg": math.degrees(_rmse(heading[steady])),
        "heading_bias_deg": math.degrees(float(np.mean(heading[steady]))),
        "cmd_w_rms_rad_s": _rmse(data["cmd_w"][steady]),
        "cmd_w_peak_rad_s": float(np.max(np.abs(data["cmd_w"][steady]))),
        "cmd_w_delta_std_rad_s": float(np.std(np.diff(data["cmd_w"][steady]))),
        "cmd_w_slew_rms_rad_s2": _rmse(dw_dt[t[1:] >= transient_s]),
        "transient_position_rmse_m": _rmse(position[~steady]),
        "transient_position_max_m": float(np.max(position[~steady])),
        "transient_heading_rmse_deg": math.degrees(_rmse(heading[~steady])),
        "transient_cmd_w_peak_rad_s": float(np.max(np.abs(data["cmd_w"][~steady]))),
    }

    stem = input_path.stem
    json_path = output_dir / f"{stem}_circle_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_path = output_dir / f"{stem}_circle_report.png"
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(9.5, 12.0))
        axes[0].plot(data["desired_x"], data["desired_y"], "k--", label="Reference")
        axes[0].plot(data["camera_x"], data["camera_y"], color="tab:blue", label="Camera")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_ylabel("y (m)")
        axes[0].legend()

        axes[1].plot(t, 100.0 * radial_error, label="Radial error")
        axes[1].plot(t, 100.0 * radial_trend, color="tab:red", label="2 s trend/bias")
        axes[1].set_ylabel("radial (cm)")
        axes[1].legend(loc="upper right")

        axes[2].plot(t, 100.0 * ripple, color="tab:orange", label="Radial waviness")
        axes[2].plot(t, np.degrees(heading), color="tab:green", alpha=0.8,
                     label="Heading error (deg)")
        axes[2].set_ylabel("ripple / heading")
        axes[2].legend(loc="upper right")

        axes[3].plot(t, data["cmd_w"], label="cmd_w")
        axes[3].plot(t, data["cmd_v"], label="cmd_v")
        axes[3].set_xlabel("active time (s)")
        axes[3].set_ylabel("command")
        axes[3].legend(loc="upper right")
        for axis in axes[1:]:
            axis.axvspan(0.0, transient_s, color="tab:gray", alpha=0.12)
        for axis in axes:
            axis.grid(True, alpha=0.3)
        fig.suptitle(
            f"Position/path RMS={100*summary['position_rmse_m']:.2f}/"
            f"{100*summary['path_rmse_m']:.2f} cm, "
            f"waviness={100*summary['radial_waviness_rmse_m']:.2f} cm, "
            f"heading RMS={summary['heading_rmse_deg']:.1f} deg, "
            f"radial bias={100*summary['radial_bias_m']:+.2f} cm"
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=220)
        plt.close(fig)
    except ImportError:
        plot_path = None
    return summary, json_path, plot_path


def _print_summary(summary, plot_path, input_path):
    print(
        "Circle report: "
        f"position/path RMS={100*summary['position_rmse_m']:.2f}/"
        f"{100*summary['path_rmse_m']:.2f} cm, "
        f"waviness RMS={100*summary['radial_waviness_rmse_m']:.2f} cm, "
        f"heading RMS={summary['heading_rmse_deg']:.1f} deg, "
        f"radial bias={100*summary['radial_bias_m']:+.2f} cm, "
        f"cmd_w RMS={summary['cmd_w_rms_rad_s']:.3f} rad/s, "
        f"plot={plot_path}"
    )
    print(f"csv={Path(input_path).expanduser().resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--transient", type=float, default=5.0)
    args = parser.parse_args(argv)
    summary, _, plot_path = analyze(
        args.input, args.output_dir or None, args.transient
    )
    _print_summary(summary, plot_path, args.input)


if __name__ == "__main__":
    main()
