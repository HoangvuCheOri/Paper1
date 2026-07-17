#!/usr/bin/env python3
"""Create a compact, straight-edge-focused report from a square run CSV."""

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


def _wrap(values):
    return np.arctan2(np.sin(values), np.cos(values))


def _rmse(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else math.nan


def _segments(
    mask, t, desired_x, desired_y, minimum_duration=2.0,
    trim_time=0.5, trim_distance=0.15,
):
    """Return established edge regions, excluding corner/recovery distance."""
    result = []
    start = None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            stop = index
            region = np.arange(start, stop)
            step = np.hypot(
                np.diff(desired_x[region]), np.diff(desired_y[region])
            )
            travelled = np.r_[0.0, np.cumsum(step)]
            keep_mask = (
                (t[region] >= t[start] + trim_time)
                & (t[region] <= t[stop - 1] - trim_time)
                & (travelled >= trim_distance)
                & (travelled <= travelled[-1] - trim_distance)
            )
            keep = region[keep_mask]
            if len(keep) >= 3 and t[keep[-1]] - t[keep[0]] >= minimum_duration:
                result.append(keep)
            start = None
    return result


def analyze(input_path, output_dir=None, warmup=5.0, max_edges=None):
    input_path = Path(input_path).expanduser().resolve()
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else input_path.parent / "square_reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = [
        "t", "camera_x", "camera_y", "camera_yaw", "desired_x", "desired_y",
        "desired_yaw", "cmd_v", "cmd_w",
    ]
    data = {key: np.asarray([_number(row, key) for row in rows]) for key in keys}
    valid = np.ones(len(rows), dtype=bool)
    for key in keys:
        valid &= np.isfinite(data[key])
    valid &= data["t"] >= warmup
    if int(valid.sum()) < 50:
        raise ValueError("CSV does not contain enough valid camera samples")
    for key in keys:
        data[key] = data[key][valid]

    t = data["t"]
    desired_yaw_unwrapped = np.unwrap(data["desired_yaw"])
    yaw_rate_reference = np.abs(np.gradient(desired_yaw_unwrapped, t))
    # The square generator uses exactly zero yaw rate on an edge. A small
    # tolerance absorbs logger timing jitter at the straight/arc transition.
    straight_mask = yaw_rate_reference < 0.12
    edge_segments = _segments(
        straight_mask, t, data["desired_x"], data["desired_y"]
    )
    if not edge_segments:
        raise ValueError("No complete straight square edge was detected")
    # An automatic run may enter the next lap briefly before shutdown. Do not
    # let that short, incomplete edge distort the straight-edge score.
    durations = np.asarray([t[idx[-1]] - t[idx[0]] for idx in edge_segments])
    typical_duration = float(np.median(durations))
    edge_segments = [
        idx for idx in edge_segments
        if t[idx[-1]] - t[idx[0]] >= 0.85 * typical_duration
    ]
    if max_edges is not None:
        edge_segments = edge_segments[:max(1, int(max_edges))]

    dx = data["desired_x"] - data["camera_x"]
    dy = data["desired_y"] - data["camera_y"]
    lateral = -np.sin(data["desired_yaw"]) * dx + np.cos(data["desired_yaw"]) * dy
    heading = _wrap(data["desired_yaw"] - data["camera_yaw"])
    position = np.hypot(dx, dy)

    edge_rows = []
    all_edge_indices = []
    for edge_number, indices in enumerate(edge_segments, 1):
        all_edge_indices.extend(indices.tolist())
        centered = lateral[indices] - np.mean(lateral[indices])
        signs = np.sign(centered)
        crossings = int(np.sum(signs[1:] * signs[:-1] < 0))
        distance = float(np.sum(np.abs(np.diff(data["desired_x"][indices]))) +
                         np.sum(np.abs(np.diff(data["desired_y"][indices]))))
        edge_rows.append({
            "edge": edge_number,
            "start_s": float(t[indices[0]]),
            "end_s": float(t[indices[-1]]),
            "duration_s": float(t[indices[-1]] - t[indices[0]]),
            "lateral_bias_m": float(np.mean(lateral[indices])),
            "lateral_waviness_rms_m": _rmse(centered),
            "lateral_peak_to_peak_m": float(np.ptp(lateral[indices])),
            "heading_rmse_deg": math.degrees(_rmse(heading[indices])),
            "cmd_w_rms_rad_s": _rmse(data["cmd_w"][indices]),
            "zero_crossings_per_m": crossings / max(distance, 0.05),
        })

    edge_indices = np.asarray(all_edge_indices, dtype=int)
    centered_edges = np.concatenate([
        lateral[idx] - np.mean(lateral[idx]) for idx in edge_segments
    ])
    summary = {
        "input": str(input_path),
        "n_complete_edges": len(edge_segments),
        "straight_lateral_bias_m": float(np.mean(lateral[edge_indices])),
        "straight_lateral_rmse_m": _rmse(lateral[edge_indices]),
        "straight_lateral_max_abs_m": float(
            np.max(np.abs(lateral[edge_indices]))
        ),
        "straight_waviness_rms_m": _rmse(centered_edges),
        "straight_lateral_peak_to_peak_max_m": max(
            row["lateral_peak_to_peak_m"] for row in edge_rows
        ),
        "straight_heading_rmse_deg": math.degrees(_rmse(heading[edge_indices])),
        "straight_cmd_w_rms_rad_s": _rmse(data["cmd_w"][edge_indices]),
        "overall_position_rmse_m": _rmse(position),
        "overall_heading_rmse_deg": math.degrees(_rmse(heading)),
    }

    stem = input_path.stem
    json_path = output_dir / f"{stem}_square_summary.json"
    csv_path = output_dir / f"{stem}_square_edges.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)

    plot_path = output_dir / f"{stem}_square_report.png"
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0))
        axes[0].plot(data["desired_x"], data["desired_y"], "k--", label="Reference")
        axes[0].plot(data["camera_x"], data["camera_y"], color="tab:blue", label="Camera")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        axes[0].legend()

        axes[1].plot(t, 100.0 * lateral, color="tab:orange", label="Lateral error")
        axes[1].axhline(0.0, color="k", linewidth=0.8)
        axes[1].set_ylabel("lateral error (cm)")
        for indices in edge_segments:
            axes[1].axvspan(t[indices[0]], t[indices[-1]], color="tab:green", alpha=0.10)
        axes[1].legend(loc="upper right")

        axes[2].plot(t, np.degrees(heading), label="Heading error (deg)")
        axes[2].plot(t, data["cmd_w"], label="cmd_w (rad/s)")
        axes[2].set_xlabel("time (s)")
        axes[2].set_ylabel("heading / angular command")
        axes[2].legend(loc="upper right")
        for axis in axes:
            axis.grid(True, alpha=0.3)
        fig.suptitle(
            f"Straight path RMS = {100*summary['straight_lateral_rmse_m']:.2f} cm, "
            f"ripple = {100*summary['straight_waviness_rms_m']:.2f} cm, "
            f"heading RMS = {summary['straight_heading_rmse_deg']:.1f} deg"
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=220)
        plt.close(fig)
    except ImportError:
        plot_path = None

    return summary, json_path, csv_path, plot_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--max-edges", type=int, default=None)
    args = parser.parse_args(argv)
    summary, json_path, csv_path, plot_path = analyze(
        args.input, args.output_dir or None, args.warmup, args.max_edges
    )
    print(
        f"Edges={summary['n_complete_edges']}, "
        f"straight path RMS={100*summary['straight_lateral_rmse_m']:.2f} cm, "
        f"straight waviness RMS={100*summary['straight_waviness_rms_m']:.2f} cm, "
        f"peak-to-peak max={100*summary['straight_lateral_peak_to_peak_max_m']:.2f} cm, "
        f"heading RMS={summary['straight_heading_rmse_deg']:.1f} deg"
    )
    print(f"Summary: {json_path}")
    print(f"Edges:   {csv_path}")
    if plot_path:
        print(f"Plot:    {plot_path}")


if __name__ == "__main__":
    main()
