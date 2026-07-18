#!/usr/bin/env python3
"""Create a tuning report focused on figure-8 path quality and transients."""

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
    """Centered moving average with reflected edges for ripple separation."""
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


def _distance_to_reference_path(x, y, reference_x, reference_y):
    """Distance from every camera point to the sampled reference polyline."""
    starts = np.column_stack((reference_x[:-1], reference_y[:-1]))
    vectors = np.diff(np.column_stack((reference_x, reference_y)), axis=0)
    lengths_sq = np.sum(vectors * vectors, axis=1)
    keep = lengths_sq > 1e-12
    starts, vectors, lengths_sq = starts[keep], vectors[keep], lengths_sq[keep]
    distances = np.empty(len(x))
    for index, point in enumerate(np.column_stack((x, y))):
        projection = np.sum((point - starts) * vectors, axis=1) / lengths_sq
        projection = np.clip(projection, 0.0, 1.0)
        closest = starts + projection[:, None] * vectors
        distances[index] = np.sqrt(np.min(np.sum((point - closest) ** 2, axis=1)))
    return distances


def _lobe_symmetry(reference_x, reference_y, actual_x, actual_y):
    """Compare first-cycle right points with mirrored left-lobe points."""
    center_x = 0.5 * (np.min(reference_x) + np.max(reference_x))
    center_y = 0.5 * (np.min(reference_y) + np.max(reference_y))
    ref_x = reference_x - center_x
    ref_y = reference_y - center_y
    act_x = actual_x - center_x
    act_y = actual_y - center_y

    left_candidates = np.flatnonzero(ref_x < -0.01)
    if not len(left_candidates):
        return math.nan, math.nan
    left_start = int(left_candidates[0])
    next_right = np.flatnonzero(
        (np.arange(len(ref_x)) > left_start) & (ref_x > 0.01)
    )
    left_end = int(next_right[0]) if len(next_right) else len(ref_x)
    right_indices = np.arange(0, left_start, 4, dtype=int)
    left_indices = np.arange(left_start, left_end, dtype=int)
    if len(right_indices) < 20 or len(left_indices) < 20:
        return math.nan, math.nan

    symmetry = []
    near_center = []
    for right_index in right_indices:
        distance_sq = (
            (ref_x[left_indices] + ref_x[right_index]) ** 2
            + (ref_y[left_indices] - ref_y[right_index]) ** 2
        )
        match_offset = int(np.argmin(distance_sq))
        if distance_sq[match_offset] > 0.02**2:
            continue
        left_index = int(left_indices[match_offset])
        mismatch = math.hypot(
            act_x[right_index] + act_x[left_index],
            act_y[right_index] - act_y[left_index],
        )
        symmetry.append(mismatch)
        if abs(ref_x[right_index]) < 0.35:
            near_center.append(mismatch)
    return _rmse(symmetry), _rmse(near_center)


def analyze(input_path, output_dir=None, transient_s=5.0):
    input_path = Path(input_path).expanduser().resolve()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir else input_path.parent / "eight_reports"
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
    # The logger retains the last message after a controller stops. Exclude
    # cached desired/command samples instead of counting the stopped tail.
    valid &= np.abs(data["ros_time"] - data["desired_stamp"]) <= 0.30
    valid &= np.abs(data["ros_time"] - data["cmd_stamp"]) <= 0.30
    if int(valid.sum()) < 100:
        raise ValueError("CSV does not contain enough fresh figure-8 samples")
    for key in keys:
        data[key] = data[key][valid]

    fresh_t = data["t"] - data["t"][0]
    moving = np.flatnonzero(data["cmd_v"] > 0.01)
    if len(moving) == 0:
        raise ValueError("CSV does not contain forward-motion samples")
    motion_start = int(moving[0])
    pre_motion_duration = float(fresh_t[motion_start])
    pre_motion_cmd_w_peak = float(
        np.max(np.abs(data["cmd_w"][:motion_start + 1]))
    )
    for key in keys:
        data[key] = data[key][motion_start:]
    t = data["t"] - data["t"][0]
    dx = data["desired_x"] - data["camera_x"]
    dy = data["desired_y"] - data["camera_y"]
    position = np.hypot(dx, dy)
    path_distance = _distance_to_reference_path(
        data["camera_x"], data["camera_y"], data["desired_x"], data["desired_y"]
    )
    lateral = -np.sin(data["desired_yaw"]) * dx + np.cos(data["desired_yaw"]) * dy
    longitudinal = (
        np.cos(data["camera_yaw"]) * dx
        + np.sin(data["camera_yaw"]) * dy
    )
    heading = _wrap(data["desired_yaw"] - data["camera_yaw"])
    symmetry, near_center_symmetry = _lobe_symmetry(
        data["desired_x"], data["desired_y"],
        data["camera_x"], data["camera_y"],
    )
    desired_yaw_rate = np.gradient(np.unwrap(data["desired_yaw"]), t)
    trend = _smooth(lateral, t)
    ripple = lateral - trend
    steady = t >= transient_s
    if int(steady.sum()) < 50:
        raise ValueError("CSV is too short after the transient interval")
    dt = np.diff(t)
    dw_dt = np.diff(data["cmd_w"]) / np.maximum(dt, 1e-3)
    negative_curve = steady & (desired_yaw_rate < -0.02)
    positive_curve = steady & (desired_yaw_rate > 0.02)
    reference_center_x = 0.5 * (
        np.min(data["desired_x"]) + np.max(data["desired_x"])
    )
    reference_center_y = 0.5 * (
        np.min(data["desired_y"]) + np.max(data["desired_y"])
    )
    center_region = steady & (
        np.hypot(
            data["desired_x"] - reference_center_x,
            data["desired_y"] - reference_center_y,
        ) <= 0.15
    )

    summary = {
        "input": str(input_path),
        "active_duration_s": float(t[-1]),
        "pre_motion_duration_s": pre_motion_duration,
        "pre_motion_cmd_w_peak_rad_s": pre_motion_cmd_w_peak,
        "n_samples": int(len(t)),
        "position_rmse_m": _rmse(position[steady]),
        "position_max_m": float(np.max(position[steady])),
        "path_rmse_m": _rmse(path_distance[steady]),
        "path_max_m": float(np.max(path_distance[steady])),
        "lateral_rmse_m": _rmse(lateral[steady]),
        "lateral_bias_m": float(np.mean(lateral[steady])),
        "lateral_bias_rms_m": _rmse(trend[steady]),
        "waviness_rmse_m": _rmse(ripple[steady]),
        "waviness_peak_to_peak_m": float(np.ptp(ripple[steady])),
        "heading_rmse_deg": math.degrees(_rmse(heading[steady])),
        "heading_bias_deg": math.degrees(float(np.mean(heading[steady]))),
        "lobe_symmetry_rmse_m": symmetry,
        "near_center_symmetry_rmse_m": near_center_symmetry,
        "center_position_rmse_m": _rmse(position[center_region]),
        "center_path_rmse_m": _rmse(path_distance[center_region]),
        "center_longitudinal_bias_m": float(np.mean(longitudinal[center_region])),
        "negative_curve_lateral_bias_m": float(np.mean(lateral[negative_curve])),
        "positive_curve_lateral_bias_m": float(np.mean(lateral[positive_curve])),
        "negative_curve_heading_bias_deg": math.degrees(
            float(np.mean(heading[negative_curve]))
        ),
        "positive_curve_heading_bias_deg": math.degrees(
            float(np.mean(heading[positive_curve]))
        ),
        "cmd_w_rms_rad_s": _rmse(data["cmd_w"][steady]),
        "cmd_w_peak_rad_s": float(np.max(np.abs(data["cmd_w"][steady]))),
        "cmd_w_saturation_fraction": float(np.mean(np.abs(data["cmd_w"][steady]) >= 0.99 * 0.85)),
        "cmd_w_delta_std_rad_s": float(np.std(np.diff(data["cmd_w"][steady]))),
        "cmd_w_slew_rms_rad_s2": _rmse(dw_dt[t[1:] >= transient_s]),
        "transient_path_rmse_m": _rmse(position[~steady]),
        "transient_path_max_m": float(np.max(position[~steady])),
        "transient_heading_rmse_deg": math.degrees(_rmse(heading[~steady])),
        "transient_cmd_w_peak_rad_s": float(np.max(np.abs(data["cmd_w"][~steady]))),
    }

    stem = input_path.stem
    json_path = output_dir / f"{stem}_eight_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_path = output_dir / f"{stem}_eight_report.png"
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(9.5, 12.0))
        axes[0].plot(data["desired_x"], data["desired_y"], "k--", label="Reference")
        axes[0].plot(data["camera_x"], data["camera_y"], color="tab:blue", label="Camera")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_ylabel("y (m)")
        axes[0].legend()

        axes[1].plot(t, 100.0 * lateral, label="Lateral error")
        axes[1].plot(t, 100.0 * trend, color="tab:red", alpha=0.8, label="2 s trend/bias")
        axes[1].set_ylabel("lateral (cm)")
        axes[1].legend(loc="upper right")

        axes[2].plot(t, 100.0 * ripple, color="tab:orange", label="Detrended waviness")
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
            f"waviness={100*summary['waviness_rmse_m']:.2f} cm, "
            f"heading RMS={summary['heading_rmse_deg']:.1f} deg, "
            f"bias={100*summary['lateral_bias_m']:+.2f} cm, "
            f"symmetry={100*summary['lobe_symmetry_rmse_m']:.2f} cm"
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=220)
        plt.close(fig)
    except ImportError:
        plot_path = None
    return summary, json_path, plot_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--transient", type=float, default=5.0)
    args = parser.parse_args(argv)
    summary, json_path, plot_path = analyze(
        args.input, args.output_dir or None, args.transient
    )
    print(
        "Eight report: "
        f"position/path RMS={100*summary['position_rmse_m']:.2f}/"
        f"{100*summary['path_rmse_m']:.2f} cm, "
        f"waviness RMS={100*summary['waviness_rmse_m']:.2f} cm, "
        f"heading RMS={summary['heading_rmse_deg']:.1f} deg, "
        f"bias={100*summary['lateral_bias_m']:+.2f} cm, "
        f"cmd_w RMS={summary['cmd_w_rms_rad_s']:.3f} rad/s, "
        f"symmetry RMS={100*summary['lobe_symmetry_rmse_m']:.2f} cm, "
        f"near-center={100*summary['near_center_symmetry_rmse_m']:.2f} cm"
        f", crossing position/path={100*summary['center_position_rmse_m']:.2f}/"
        f"{100*summary['center_path_rmse_m']:.2f} cm"
    )
    print(f"Summary: {json_path}")
    if plot_path:
        print(f"plot={plot_path}")
    print(f"csv={Path(args.input).expanduser().resolve()}")


if __name__ == "__main__":
    main()
