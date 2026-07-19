#!/usr/bin/env python3
"""Regenerate comparison figures/tables whenever a hardware run finishes."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def _save(fig, output, stem):
    paths = []
    for extension, options in (("pdf", {}), ("png", {"dpi": 600})):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", **options)
        paths.append(path)
    return paths


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.7, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def _local_arrays(rows, figure_rotation_deg=0.0):
    import numpy as np
    keys = (
        "t", "ros_time", "desired_x", "desired_y", "desired_yaw",
        "odom_x", "odom_y", "camera_x", "camera_y", "camera_yaw",
        "camera_stamp", "camera_age_s",
        "camera_error_ex", "camera_error_ey", "camera_error_etheta", "cmd_v",
    )
    values = {key: np.asarray([_float(row, key) for row in rows]) for key in keys}
    if not np.isfinite(values["camera_age_s"]).any():
        values["camera_age_s"] = values["ros_time"] - values["camera_stamp"]
    if not np.isfinite(values["camera_error_ex"]).any():
        dx = values["desired_x"] - values["camera_x"]
        dy = values["desired_y"] - values["camera_y"]
        yaw = values["camera_yaw"]
        values["camera_error_ex"] = np.cos(yaw) * dx + np.sin(yaw) * dy
        values["camera_error_ey"] = -np.sin(yaw) * dx + np.cos(yaw) * dy
        heading = values["desired_yaw"] - yaw
        values["camera_error_etheta"] = np.arctan2(np.sin(heading), np.cos(heading))
        fresh = np.isfinite(values["camera_age_s"]) & (values["camera_age_s"] <= 0.30)
        for key in ("camera_error_ex", "camera_error_ey", "camera_error_etheta"):
            values[key][~fresh] = np.nan
    valid = np.flatnonzero(
        np.isfinite(values["desired_x"]) & np.isfinite(values["desired_y"])
        & np.isfinite(values["desired_yaw"])
    )
    if not valid.size:
        return values
    first = valid[0]
    x0, y0, yaw0 = (
        values["desired_x"][first], values["desired_y"][first],
        values["desired_yaw"][first],
    )
    cosine, sine = math.cos(yaw0), math.sin(yaw0)
    for prefix in ("desired", "odom", "camera"):
        dx = values[f"{prefix}_x"] - x0
        dy = values[f"{prefix}_y"] - y0
        values[f"{prefix}_x_local"] = cosine * dx + sine * dy
        values[f"{prefix}_y_local"] = -sine * dx + cosine * dy
    if abs(float(figure_rotation_deg)) > 1e-12:
        angle = math.radians(float(figure_rotation_deg))
        frame_cosine, frame_sine = math.cos(angle), math.sin(angle)
        for prefix in ("desired", "odom", "camera"):
            x_values = values[f"{prefix}_x_local"].copy()
            y_values = values[f"{prefix}_y_local"].copy()
            values[f"{prefix}_x_local"] = (
                frame_cosine * x_values - frame_sine * y_values
            )
            values[f"{prefix}_y_local"] = (
                frame_sine * x_values + frame_cosine * y_values
            )
    return values


def _controller_summaries(root):
    summaries = []
    for path in Path(root).glob("*_summary.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (value.get("controller") in ("BSMC", "Backstepping")
                and value.get("csv") and value.get("valid", True)
                and int(value.get("n_metric", 3)) >= 3):
            value["summary_path"] = str(path)
            summaries.append(value)
    return summaries


def refresh_tracking_assets(root):
    """Create paired trajectory/error figures and repeat-aware tracking tables."""
    import numpy as np
    plt = _setup_matplotlib()
    root = Path(root)
    output = root / "publication"
    output.mkdir(parents=True, exist_ok=True)
    summaries = _controller_summaries(root)
    grouped = defaultdict(list)
    for summary in summaries:
        grouped[(summary["trajectory"], summary["controller"])].append(summary)

    outputs = []
    for trajectory in ("circle", "eight", "square"):
        baseline = grouped.get((trajectory, "Backstepping"), [])
        compensated = grouped.get((trajectory, "BSMC"), [])
        if not baseline or not compensated:
            continue
        baseline_run = max(baseline, key=lambda item: item["summary_path"])
        compensated_run = max(compensated, key=lambda item: item["summary_path"])
        pair = {}
        selected_runs = {
            "Backstepping": baseline_run,
            "BSMC": compensated_run,
        }
        for controller, selected_run in selected_runs.items():
            path_rotation = float(
                selected_run.get("parameters", {}).get("path_rotation_deg", 0.0)
            )
            frame_rotation = -path_rotation if trajectory == "eight" else 0.0
            pair[controller] = _local_arrays(
                _read_csv(selected_run["csv"]),
                figure_rotation_deg=frame_rotation,
            )

        fig, ax = plt.subplots(figsize=(3.5, 3.15), constrained_layout=True)
        reference = pair["BSMC"]
        ax.plot(reference["desired_x_local"], reference["desired_y_local"],
                "--", color="#222222", lw=1.5, label="Reference")
        colors = {"Backstepping": "#E69F00", "BSMC": "#0072B2"}
        labels = {
            "Backstepping": "Nominal Backstepping (EKF)",
            "BSMC": "Compensated BSMC (EKF)",
        }
        for controller in ("Backstepping", "BSMC"):
            data = pair[controller]
            alpha = 0.82 if trajectory == "eight" else 0.95
            ax.plot(data["odom_x_local"], data["odom_y_local"],
                    color=colors[controller], alpha=alpha, label=labels[controller])
            camera_ok = (
                np.isfinite(data["camera_x_local"]) & np.isfinite(data["camera_y_local"])
                & np.isfinite(data["camera_age_s"]) & (data["camera_age_s"] <= 0.30)
            )
            ax.scatter(data["camera_x_local"][camera_ok], data["camera_y_local"][camera_ok],
                       s=2.5, color=colors[controller], alpha=0.12, edgecolors="none",
                       label="Raw AprilTag pose" if controller == "BSMC" else None)
        valid_ref = np.flatnonzero(
            np.isfinite(reference["desired_x_local"]) & np.isfinite(reference["desired_y_local"])
        )
        if valid_ref.size:
            start = valid_ref[0]
            ax.plot(reference["desired_x_local"][start], reference["desired_y_local"][start],
                    marker="^", ms=6, color="#009E73", linestyle="none", label="Start")
            ax.annotate("Start", (reference["desired_x_local"][start], reference["desired_y_local"][start]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        if trajectory == "square":
            x_ref, y_ref = reference["desired_x_local"], reference["desired_y_local"]
            finite = np.isfinite(x_ref) & np.isfinite(y_ref)
            if finite.any():
                xmin, xmax = np.nanmin(x_ref[finite]), np.nanmax(x_ref[finite])
                ymin, ymax = np.nanmin(y_ref[finite]), np.nanmax(y_ref[finite])
                ax.scatter([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax],
                           marker="s", s=15, facecolors="white", edgecolors="#222222", zorder=4)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.legend(loc="best", frameon=True)
        outputs += _save(fig, output, f"{trajectory}_controller_comparison")
        plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(7.16, 4.65), sharex="col", constrained_layout=True)
        error_keys = ("camera_error_ex", "camera_error_ey", "camera_error_etheta")
        ylabels = (r"$e_x$ (m)", r"$e_y$ (m)", r"$e_\theta$ (rad)")
        for column, controller in enumerate(("Backstepping", "BSMC")):
            data = pair[controller]
            axes[0, column].set_title("Nominal Backstepping" if column == 0 else "Compensated BSMC")
            for row_index, (key, ylabel) in enumerate(zip(error_keys, ylabels)):
                axes[row_index, column].plot(data["t"], data[key], color=colors[controller], lw=0.8)
                axes[row_index, column].axhline(0.0, color="#555555", ls="--", lw=0.5)
                axes[row_index, column].grid(True, alpha=0.3, lw=0.5)
                if column == 0:
                    axes[row_index, column].set_ylabel(ylabel)
            axes[-1, column].set_xlabel("time (s)")
        for row_index, key in enumerate(error_keys):
            combined = np.concatenate((pair["Backstepping"][key], pair["BSMC"][key]))
            finite = combined[np.isfinite(combined)]
            if finite.size:
                bound = max(np.max(np.abs(finite)) * 1.05, 0.01)
                axes[row_index, 0].set_ylim(-bound, bound)
                axes[row_index, 1].set_ylim(-bound, bound)
        outputs += _save(fig, output, f"{trajectory}_error_comparison")
        plt.close(fig)

        provenance = {
            "trajectory": trajectory,
            "selection": "latest completed run for each controller",
            "backstepping_run_id": baseline_run["run_id"],
            "bsmc_run_id": compensated_run["run_id"],
            "backstepping_csv": baseline_run["csv"],
            "bsmc_csv": compensated_run["csv"],
            "camera_freshness_limit_s": 0.30,
            "figure_frame": "trajectory-aligned for figure-eight; local start frame otherwise",
            "edits": "declared rigid coordinate translation/rotation only; no smoothing",
        }
        path = output / f"{trajectory}_comparison_provenance.json"
        path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        outputs.append(path)

    table_rows = []
    for (trajectory, controller), runs in sorted(grouped.items()):
        position = [run.get("camera_rmse_position_m", math.nan) for run in runs]
        heading = [run.get("camera_rmse_heading_deg", math.nan) for run in runs]
        position = np.asarray(position, dtype=float)
        heading = np.asarray(heading, dtype=float)
        position = position[np.isfinite(position)]
        heading = heading[np.isfinite(heading)]
        if not position.size:
            continue
        table_rows.append({
            "trajectory": trajectory, "controller": controller,
            "n": int(position.size),
            "position_rmse_mean_cm": f"{100.0 * np.mean(position):.2f}",
            "position_rmse_std_cm": f"{100.0 * np.std(position, ddof=1):.2f}" if position.size > 1 else "--",
            "heading_rmse_mean_deg": f"{np.mean(heading):.2f}" if heading.size else "--",
            "heading_rmse_std_deg": f"{np.std(heading, ddof=1):.2f}" if heading.size > 1 else "--",
        })
    if table_rows:
        csv_path = output / "table_tracking_repeats.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
        tex_path = output / "table_tracking_repeats.tex"
        lines = [
            "\\begin{table}[!t]", "\\centering",
            "\\caption{Camera-derived tracking performance across independent hardware runs.}",
            "\\label{tab:tracking}", "\\begin{tabular}{llrrr}", "\\hline",
            "Trajectory & Controller & $n$ & Position (cm) & Heading (deg) \\\\", "\\hline",
        ]
        for row in table_rows:
            position = row["position_rmse_mean_cm"]
            heading = row["heading_rmse_mean_deg"]
            if row["position_rmse_std_cm"] != "--":
                position += "$\\pm$" + row["position_rmse_std_cm"]
            if row["heading_rmse_std_deg"] != "--":
                heading += "$\\pm$" + row["heading_rmse_std_deg"]
            lines.append(
                f"{row['trajectory']} & {row['controller']} & {row['n']} & {position} & {heading} \\\\"
            )
        lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
        tex_path.write_text("\n".join(lines), encoding="utf-8")
        outputs += [csv_path, tex_path]
    return outputs


def refresh_link_assets(root, nominal_ms=50.0):
    """Create the multi-condition ESP-NOW distribution and numeric table."""
    import numpy as np
    plt = _setup_matplotlib()
    root = Path(root)
    output = root / "publication"
    output.mkdir(parents=True, exist_ok=True)
    raw_groups = defaultdict(list)
    for path in root.glob("*_summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "packets_received" not in summary:
            continue
        csv_name = path.name.replace("_summary.json", ".csv")
        csv_path = root / csv_name
        if not csv_path.exists():
            continue
        rows = _read_csv(csv_path)
        values = np.asarray([_float(row, "interarrival_ms") for row in rows])
        values = values[np.isfinite(values) & (values >= 0.0)]
        if values.size:
            key = (str(summary["condition"]), float(summary["distance_m"]))
            raw_groups[key].append((summary, values))
    if not raw_groups:
        return []
    groups = []
    for (condition, distance), repeats in raw_groups.items():
        values = np.concatenate([item[1] for item in repeats])
        received = sum(int(item[0]["packets_received"]) for item in repeats)
        missing = sum(int(item[0]["missing_packets"]) for item in repeats)
        methods = {item[0]["loss_method"] for item in repeats}
        summary = {
            "condition": condition,
            "distance_m": distance,
            "n_runs": len(repeats),
            "packets_received": received,
            "missing_packets": missing,
            "loss_percent": 100.0 * missing / max(1, received + missing),
            "loss_method": (
                next(iter(methods)) if len(methods) == 1
                else "mixed; inspect run summaries"
            ),
        }
        groups.append((summary, values))
    groups.sort(key=lambda item: (item[0]["distance_m"], item[0]["condition"]))
    labels = [f"{summary['condition'].title()}\n{summary['distance_m']:g} m" for summary, _ in groups]
    fig, ax = plt.subplots(figsize=(7.16, 3.0), constrained_layout=True)
    boxes = ax.boxplot([values for _, values in groups], labels=labels, patch_artist=True,
                       showfliers=True, widths=0.55,
                       medianprops={"color": "#222222", "linewidth": 1.1},
                       flierprops={"marker": ".", "markersize": 2.2, "alpha": 0.3})
    for index, box in enumerate(boxes["boxes"]):
        box.set_facecolor("#0072B2" if "moving" in groups[index][0]["condition"].lower() else "#E69F00")
        box.set_alpha(0.65)
    ax.axhline(nominal_ms, color="#222222", ls="--", lw=0.8, label=f"Nominal {nominal_ms:g} ms")
    ymax = ax.get_ylim()[1]
    for index, (summary, _) in enumerate(groups, 1):
        ax.text(index, ymax, f"{summary['loss_percent']:.2f}%", ha="center", va="top", fontsize=7)
    ax.set_xlabel("condition")
    ax.set_ylabel("inter-arrival time (ms)")
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.legend(loc="best")
    outputs = _save(fig, output, "fig6_espnow_interarrival")
    plt.close(fig)

    rows_out = []
    for summary, values in groups:
        rows_out.append({
            "condition": summary["condition"], "distance_m": summary["distance_m"],
            "n_runs": summary["n_runs"], "packets": summary["packets_received"],
            "median_ms": f"{np.median(values):.2f}",
            "p95_ms": f"{np.percentile(values, 95):.2f}",
            "p99_ms": f"{np.percentile(values, 99):.2f}",
            "loss_percent": f"{summary['loss_percent']:.2f}",
            "loss_method": summary["loss_method"],
        })
    csv_path = output / "table_espnow.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    tex_path = output / "table_espnow.tex"
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{ESP-NOW packet inter-arrival and loss measurements. Loss uses sequence numbers where available; otherwise it is gap-estimated.}",
        "\\label{tab:espnow}", "\\begin{tabular}{lrrrrr}", "\\hline",
        "Condition & $n$ & Median & P95 & P99 & Loss (\\%) \\\\", "\\hline",
    ]
    for row in rows_out:
        condition = f"{row['condition']} {row['distance_m']} m"
        lines.append(
            f"{condition} & {row['n_runs']} & {row['median_ms']} & {row['p95_ms']} & {row['p99_ms']} & {row['loss_percent']} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return outputs + [csv_path, tex_path]
