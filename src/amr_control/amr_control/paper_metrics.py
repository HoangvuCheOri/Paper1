#!/usr/bin/env python3
"""Offline metrics and plots for the BSMC paper."""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv_rows(path):
    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = []
        for row in reader:
            rows.append({str(k).strip().lower(): v for k, v in row.items()})
    return rows


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def first_value(rows, aliases, default=""):
    aliases = [alias.lower() for alias in aliases]
    for row in rows:
        for alias in aliases:
            value = row.get(alias, "")
            if str(value).strip():
                return value
    return default


def column(rows, aliases):
    aliases = [alias.lower() for alias in aliases]
    values = []
    for row in rows:
        value = np.nan
        for alias in aliases:
            if alias in row:
                value = safe_float(row.get(alias))
                break
        values.append(value)
    return np.asarray(values, dtype=float)


def text_column(rows, aliases, default=""):
    aliases = [alias.lower() for alias in aliases]
    values = []
    for row in rows:
        value = default
        for alias in aliases:
            if alias in row and str(row.get(alias)).strip():
                value = str(row.get(alias)).strip()
                break
        values.append(value)
    return values


def finite_any(values):
    return bool(np.isfinite(values).any())


def angle_column(rows, rad_aliases, deg_aliases):
    rad = column(rows, rad_aliases)
    if finite_any(rad):
        return rad
    deg = column(rows, deg_aliases)
    return np.deg2rad(deg)


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def rmse(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.nanmean(values * values)))


def nanmean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return np.nan
    return float(np.nanmean(values))


def nanstd(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size <= 1:
        return 0.0 if finite.size == 1 else np.nan
    return float(np.std(finite, ddof=1))


def sanitize_name(value):
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return safe.strip("_") or "run"


def pose_columns(rows, source):
    source = source.lower()
    if source == "camera":
        x = column(rows, ["camera_x", "cam_x"])
        y = column(rows, ["camera_y", "cam_y"])
        yaw = angle_column(rows, ["camera_yaw", "cam_yaw"], ["camera_yaw(deg)", "cam_yaw(deg)"])
    else:
        x = column(rows, ["odom_x"])
        y = column(rows, ["odom_y"])
        yaw = angle_column(rows, ["odom_yaw"], ["odom_yaw(deg)"])
    return x, y, yaw


def convergence_time(t, ep, epsilon, hold_time):
    if len(t) == 0:
        return np.nan
    good = np.asarray(ep) <= epsilon
    for idx in range(len(t)):
        end_idx = int(np.searchsorted(t, t[idx] + hold_time, side="right"))
        if end_idx <= idx:
            continue
        if bool(np.all(good[idx:end_idx])):
            return float(t[idx])
    return np.nan


def compute_tracking_metrics(
    rows,
    source="odom",
    start_time=0.0,
    end_time=None,
    epsilon=0.05,
    hold_time=2.0,
):
    t = column(rows, ["t", "time"])
    if not finite_any(t):
        raise ValueError("CSV has no usable time column (expected t or Time).")

    actual_x, actual_y, actual_yaw = pose_columns(rows, source)
    desired_x = column(rows, ["desired_x"])
    desired_y = column(rows, ["desired_y"])
    desired_yaw = angle_column(rows, ["desired_yaw"], ["desired_yaw(deg)"])

    mask = (
        np.isfinite(t)
        & np.isfinite(actual_x)
        & np.isfinite(actual_y)
        & np.isfinite(actual_yaw)
        & np.isfinite(desired_x)
        & np.isfinite(desired_y)
        & np.isfinite(desired_yaw)
        & (t >= start_time)
    )
    if end_time is not None:
        mask &= t <= end_time
    if int(mask.sum()) < 3:
        raise ValueError("Not enough valid tracking samples after filtering.")

    t = t[mask]
    actual_x = actual_x[mask]
    actual_y = actual_y[mask]
    actual_yaw = actual_yaw[mask]
    desired_x = desired_x[mask]
    desired_y = desired_y[mask]
    desired_yaw = desired_yaw[mask]

    dx = desired_x - actual_x
    dy = desired_y - actual_y
    ep = np.hypot(dx, dy)

    direct_ex = column(rows, ["error_ex"])[mask]
    direct_ey = column(rows, ["error_ey"])[mask]
    direct_etheta = column(rows, ["error_etheta"])[mask]
    use_direct_error = source.lower() == "odom" and finite_any(direct_ex)

    if use_direct_error:
        ex = direct_ex
        ey = direct_ey
        etheta = wrap_angle(direct_etheta)
    else:
        ex = np.cos(actual_yaw) * dx + np.sin(actual_yaw) * dy
        ey = -np.sin(actual_yaw) * dx + np.cos(actual_yaw) * dy
        etheta = wrap_angle(desired_yaw - actual_yaw)

    cmd_v = column(rows, ["cmd_v"])[mask]
    cmd_w = column(rows, ["cmd_w"])[mask]

    metrics = {
        "n_samples": int(len(t)),
        "duration_s": float(t[-1] - t[0]),
        "rmse_ex_m": rmse(ex),
        "rmse_ey_m": rmse(ey),
        "rmse_etheta_rad": rmse(etheta),
        "rmse_position_m": rmse(ep),
        "mae_position_m": nanmean(np.abs(ep)),
        "max_position_m": float(np.nanmax(ep)),
        "mae_heading_rad": nanmean(np.abs(etheta)),
        "max_heading_rad": float(np.nanmax(np.abs(etheta))),
        "convergence_time_s": convergence_time(t, ep, epsilon, hold_time),
        "mean_abs_cmd_v": nanmean(np.abs(cmd_v)),
        "mean_abs_cmd_w": nanmean(np.abs(cmd_w)),
    }
    arrays = {
        "t": t,
        "actual_x": actual_x,
        "actual_y": actual_y,
        "actual_yaw": actual_yaw,
        "desired_x": desired_x,
        "desired_y": desired_y,
        "desired_yaw": desired_yaw,
        "ex": ex,
        "ey": ey,
        "etheta": etheta,
        "ep": ep,
    }
    return metrics, arrays


def plot_tracking(arrays, outdir, name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    paths = []
    xy_path = os.path.join(outdir, f"{name}_xy.png")
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(arrays["desired_x"], arrays["desired_y"], "k--", label="Reference")
    plt.plot(arrays["actual_x"], arrays["actual_y"], "b", label="Actual")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(xy_path, dpi=300)
    plt.close()
    paths.append(xy_path)

    err_path = os.path.join(outdir, f"{name}_errors.png")
    plt.figure(figsize=(6.0, 4.2))
    plt.plot(arrays["t"], arrays["ex"], label="e_x (m)")
    plt.plot(arrays["t"], arrays["ey"], label="e_y (m)")
    plt.plot(arrays["t"], arrays["etheta"], label="e_theta (rad)")
    plt.plot(arrays["t"], arrays["ep"], "k--", linewidth=1.0, label="e_p (m)")
    plt.grid(True, alpha=0.3)
    plt.xlabel("time (s)")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(err_path, dpi=300)
    plt.close()
    paths.append(err_path)
    return paths


def manifest_runs(manifest_path):
    base = Path(manifest_path).resolve().parent
    rows = read_csv_rows(manifest_path)
    runs = []
    for row in rows:
        file_value = row.get("file") or row.get("path") or row.get("input")
        if not file_value:
            continue
        path = Path(file_value)
        if not path.is_absolute():
            path = base / path
        runs.append(
            {
                "file": str(path),
                "controller": row.get("controller", ""),
                "trajectory": row.get("trajectory", ""),
                "run_id": row.get("run_id", ""),
                "source": row.get("source", ""),
                "start_time": safe_float(row.get("start_time"), np.nan),
                "end_time": safe_float(row.get("end_time"), np.nan),
            }
        )
    return runs


def tracking_command(args):
    ensure_dir(args.outdir)
    runs = []
    if args.manifest:
        runs.extend(manifest_runs(args.manifest))
    for input_path in args.input or []:
        runs.append(
            {
                "file": input_path,
                "controller": args.controller,
                "trajectory": args.trajectory,
                "run_id": args.run_id,
                "source": args.source,
                "start_time": args.start_time,
                "end_time": args.end_time,
            }
        )
    if not runs:
        raise SystemExit("No tracking inputs. Use --input or --manifest.")

    run_rows = []
    plotted = []
    for run in runs:
        rows = read_csv_rows(run["file"])
        controller = run["controller"] or first_value(rows, ["controller"], "unknown")
        trajectory = run["trajectory"] or first_value(rows, ["trajectory"], "unknown")
        run_id = run["run_id"] or first_value(rows, ["run_id"], Path(run["file"]).stem)
        source = run["source"] or args.source
        start_time = run["start_time"]
        if start_time is None or not np.isfinite(start_time):
            start_time = args.start_time
        end_time = run["end_time"]
        if end_time is None or not np.isfinite(end_time):
            end_time = args.end_time
        if end_time is not None and not np.isfinite(end_time):
            end_time = None

        metrics, arrays = compute_tracking_metrics(
            rows,
            source=source,
            start_time=start_time,
            end_time=end_time,
            epsilon=args.epsilon,
            hold_time=args.hold_time,
        )
        output_row = {
            "file": run["file"],
            "controller": controller,
            "trajectory": trajectory,
            "run_id": run_id,
            "source": source,
            "start_time_s": start_time,
            "end_time_s": "" if end_time is None else end_time,
        }
        output_row.update(metrics)
        run_rows.append(output_row)

        if not args.no_plots:
            name = sanitize_name(f"{trajectory}_{controller}_{run_id}_{source}")
            plotted.extend(plot_tracking(arrays, args.outdir, name))

    run_fields = [
        "file",
        "controller",
        "trajectory",
        "run_id",
        "source",
        "start_time_s",
        "end_time_s",
        "n_samples",
        "duration_s",
        "rmse_ex_m",
        "rmse_ey_m",
        "rmse_etheta_rad",
        "rmse_position_m",
        "mae_position_m",
        "max_position_m",
        "mae_heading_rad",
        "max_heading_rad",
        "convergence_time_s",
        "mean_abs_cmd_v",
        "mean_abs_cmd_w",
    ]
    write_csv(os.path.join(args.outdir, "tracking_runs.csv"), run_rows, run_fields)

    metric_names = [
        "rmse_ex_m",
        "rmse_ey_m",
        "rmse_etheta_rad",
        "rmse_position_m",
        "mae_position_m",
        "max_position_m",
        "convergence_time_s",
    ]
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["trajectory"], row["controller"], row["source"])].append(row)

    summary_rows = []
    for (trajectory, controller, source), group in sorted(grouped.items()):
        summary = {
            "trajectory": trajectory,
            "controller": controller,
            "source": source,
            "n_runs": len(group),
        }
        for metric in metric_names:
            values = [safe_float(row.get(metric)) for row in group]
            summary[f"{metric}_mean"] = nanmean(values)
            summary[f"{metric}_std"] = nanstd(values)
        summary_rows.append(summary)
    summary_fields = ["trajectory", "controller", "source", "n_runs"]
    for metric in metric_names:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(
        os.path.join(args.outdir, "tracking_summary.csv"),
        summary_rows,
        summary_fields,
    )
    write_tracking_latex(
        os.path.join(args.outdir, "tracking_table_latex.tex"), summary_rows
    )
    print(f"Wrote {len(run_rows)} run rows to {args.outdir}/tracking_runs.csv")
    if plotted:
        print(f"Wrote {len(plotted)} tracking plots to {args.outdir}")


def fmt_mean_std(row, metric, digits=3):
    mean = safe_float(row.get(f"{metric}_mean"))
    std = safe_float(row.get(f"{metric}_std"))
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_tracking_latex(path, summary_rows):
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Trajectory & Controller & RMSE $e_x$ & RMSE $e_y$ & RMSE $e_\\theta$ & Max $e_p$ \\\\",
        "\\hline",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['trajectory']} & {row['controller']} & "
            f"{fmt_mean_std(row, 'rmse_ex_m')} & "
            f"{fmt_mean_std(row, 'rmse_ey_m')} & "
            f"{fmt_mean_std(row, 'rmse_etheta_rad')} & "
            f"{fmt_mean_std(row, 'max_position_m')} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    with open(path, "w") as tex_file:
        tex_file.write("\n".join(lines))


def first_crossing(t, y, target, start_idx, sign):
    values = sign * (y[start_idx:] - target)
    indices = np.flatnonzero(values >= 0.0)
    if indices.size == 0:
        return None
    return int(start_idx + indices[0])


def analyze_step(rows, kind, command_col, response_col):
    t = column(rows, ["t", "time"])
    cmd = column(rows, [command_col])
    response = column(rows, [response_col])
    mask = np.isfinite(t) & np.isfinite(cmd) & np.isfinite(response)
    t = t[mask]
    cmd = cmd[mask]
    response = response[mask]
    if len(t) < 10:
        raise ValueError("Not enough valid samples for step analysis.")

    n = len(t)
    head = max(3, n // 10)
    tail_start = max(0, int(0.8 * n))
    cmd_initial = float(np.nanmedian(cmd[:head]))
    cmd_final = float(np.nanmedian(cmd[tail_start:]))
    cmd_delta = cmd_final - cmd_initial
    if abs(cmd_delta) < 1e-9:
        raise ValueError("Command column does not contain a detectable step.")
    sign = 1.0 if cmd_delta >= 0.0 else -1.0
    step_mid = cmd_initial + 0.5 * cmd_delta
    step_candidates = np.flatnonzero(sign * (cmd - step_mid) >= 0.0)
    if step_candidates.size == 0:
        raise ValueError("Could not find command step start.")
    step_idx = int(step_candidates[0])
    step_time = float(t[step_idx])

    response_initial = float(np.nanmedian(response[: max(1, step_idx)]))
    response_final = float(np.nanmedian(response[tail_start:]))
    response_delta = response_final - response_initial
    if abs(response_delta) < 1e-9:
        raise ValueError("Response column does not change enough.")
    response_sign = 1.0 if response_delta >= 0.0 else -1.0

    target63 = response_initial + 0.632 * response_delta
    idx63 = first_crossing(t, response, target63, step_idx, response_sign)
    tau = float(t[idx63] - step_time) if idx63 is not None else np.nan
    bandwidth = 1.0 / tau if np.isfinite(tau) and tau > 0.0 else np.nan

    target10 = response_initial + 0.10 * response_delta
    target90 = response_initial + 0.90 * response_delta
    idx10 = first_crossing(t, response, target10, step_idx, response_sign)
    idx90 = first_crossing(t, response, target90, step_idx, response_sign)
    rise_time = (
        float(t[idx90] - t[idx10])
        if idx10 is not None and idx90 is not None and idx90 >= idx10
        else np.nan
    )

    peak = np.nanmax(response[step_idx:]) if response_sign > 0 else np.nanmin(response[step_idx:])
    overshoot = response_sign * (peak - response_final) / abs(response_delta) * 100.0
    overshoot = max(0.0, float(overshoot))

    band = 0.05 * abs(response_delta)
    settling_time = np.nan
    for idx in range(step_idx, len(t)):
        if np.all(np.abs(response[idx:] - response_final) <= band):
            settling_time = float(t[idx] - step_time)
            break

    return {
        "kind": kind,
        "command_col": command_col,
        "response_col": response_col,
        "n_samples": len(t),
        "step_time_s": step_time,
        "command_initial": cmd_initial,
        "command_final": cmd_final,
        "response_initial": response_initial,
        "response_final": response_final,
        "tau_63_s": tau,
        "bandwidth_1_per_s": bandwidth,
        "rise_time_10_90_s": rise_time,
        "settling_time_5pct_s": settling_time,
        "overshoot_percent": overshoot,
    }, {"t": t, "cmd": cmd, "response": response, "step_time": step_time}


def plot_step(arrays, outdir, name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    path = os.path.join(outdir, f"{name}_step.png")
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(arrays["t"], arrays["cmd"], "k--", label="command")
    plt.plot(arrays["t"], arrays["response"], "b", label="response")
    plt.axvline(arrays["step_time"], color="r", linewidth=1.0, alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel("time (s)")
    plt.ylabel("signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def step_command(args):
    ensure_dir(args.outdir)
    rows = read_csv_rows(args.input)
    command_col = args.command_col or ("cmd_v" if args.kind == "linear" else "cmd_w")
    response_col = args.response_col or ("odom_v" if args.kind == "linear" else "odom_w")
    metrics, arrays = analyze_step(rows, args.kind, command_col, response_col)
    metrics["file"] = args.input
    path = os.path.join(args.outdir, f"step_{sanitize_name(args.kind)}_summary.csv")
    write_csv(path, [metrics])
    if not args.no_plots:
        plot_step(arrays, args.outdir, sanitize_name(args.kind))
    print(f"Wrote step summary to {path}")


def unique_link_samples(rows):
    rx_time = column(rows, ["rx_time", "espnow_rx_time", "ros_time", "t", "time"])
    seq = column(rows, ["seq", "espnow_seq"])
    keep = []
    seen_seq = set()
    seen_rx = set()
    for idx, (rx, sequence) in enumerate(zip(rx_time, seq)):
        if not np.isfinite(rx):
            continue
        if np.isfinite(sequence):
            key = int(sequence)
            if key in seen_seq:
                continue
            seen_seq.add(key)
        else:
            key = round(float(rx), 6)
            if key in seen_rx:
                continue
            seen_rx.add(key)
        keep.append(idx)
    return keep


def analyze_link(rows, nominal_period):
    keep = unique_link_samples(rows)
    if len(keep) < 3:
        raise ValueError("Not enough packet-level link samples.")

    rx = column(rows, ["rx_time", "espnow_rx_time", "ros_time", "t", "time"])[keep]
    seq = column(rows, ["seq", "espnow_seq"])[keep]
    interarrival = column(rows, ["interarrival_ms", "espnow_interarrival_ms"])[keep]

    if not finite_any(interarrival):
        interarrival = np.diff(rx, prepend=np.nan) * 1000.0
    interarrival = interarrival[np.isfinite(interarrival)]
    if interarrival.size == 0:
        raise ValueError("No valid inter-arrival samples.")

    median_period_ms = float(np.nanmedian(interarrival))
    jitter95_ms = float(np.nanpercentile(np.abs(interarrival - median_period_ms), 95))
    duration_s = float(np.nanmax(rx) - np.nanmin(rx))
    packet_rate_hz = float(len(keep) / duration_s) if duration_s > 0.0 else np.nan

    finite_seq = seq[np.isfinite(seq)]
    if finite_seq.size >= 2:
        expected = int(np.nanmax(finite_seq) - np.nanmin(finite_seq) + 1)
        missing = max(0, expected - len(np.unique(finite_seq.astype(int))))
    else:
        expected_gaps = np.maximum(
            0.0, np.round(interarrival / (nominal_period * 1000.0)) - 1.0
        )
        missing = int(np.nansum(expected_gaps))
        expected = len(keep) + missing
    loss_percent = 100.0 * missing / expected if expected > 0 else np.nan

    latency = column(rows, ["latency_ms"])[keep]
    median_latency_ms = float(np.nanmedian(latency)) if finite_any(latency) else np.nan

    return {
        "n_packets": len(keep),
        "duration_s": duration_s,
        "packet_rate_hz": packet_rate_hz,
        "median_period_ms": median_period_ms,
        "jitter95_ms": jitter95_ms,
        "missing_packets": missing,
        "loss_percent": loss_percent,
        "median_latency_ms": median_latency_ms,
    }


def link_command(args):
    ensure_dir(args.outdir)
    rows = read_csv_rows(args.input)
    metrics = analyze_link(rows, args.nominal_period)
    metrics["file"] = args.input
    metrics["condition"] = args.condition or first_value(rows, ["condition"], "")
    distance = args.distance_m
    if not np.isfinite(distance):
        distance = safe_float(first_value(rows, ["distance_m"], ""), np.nan)
    metrics["distance_m"] = distance
    path = os.path.join(args.outdir, "link_quality_summary.csv")
    write_csv(path, [metrics])
    print(f"Wrote link summary to {path}")


def localization_command(args):
    ensure_dir(args.outdir)
    rows = read_csv_rows(args.input)
    t = column(rows, ["t", "time"])
    odom_x, odom_y, odom_yaw = pose_columns(rows, "odom")
    cam_x, cam_y, cam_yaw = pose_columns(rows, "camera")
    dx = column(rows, ["diff_x"])
    dy = column(rows, ["diff_y"])
    dyaw_deg = column(rows, ["diff_yaw"])
    if not finite_any(dx):
        dx = cam_x - odom_x
    if not finite_any(dy):
        dy = cam_y - odom_y
    if finite_any(dyaw_deg):
        dyaw = np.deg2rad(dyaw_deg)
    else:
        dyaw = wrap_angle(cam_yaw - odom_yaw)

    mask = np.isfinite(t) & np.isfinite(dx) & np.isfinite(dy)
    if args.start_time is not None:
        mask &= t >= args.start_time
    if args.end_time is not None:
        mask &= t <= args.end_time
    if int(mask.sum()) < 3:
        raise ValueError("Not enough odom/camera samples.")

    t = t[mask]
    dx = dx[mask]
    dy = dy[mask]
    dyaw = dyaw[mask]
    drift = np.hypot(dx, dy)
    row = {
        "file": args.input,
        "n_samples": len(t),
        "duration_s": float(t[-1] - t[0]),
        "rmse_dx_m": rmse(dx),
        "rmse_dy_m": rmse(dy),
        "rmse_position_m": rmse(drift),
        "mae_position_m": nanmean(np.abs(drift)),
        "max_position_m": float(np.nanmax(drift)),
        "final_position_m": float(drift[-1]),
        "rmse_yaw_rad": rmse(dyaw),
        "max_yaw_rad": float(np.nanmax(np.abs(dyaw))),
    }
    path = os.path.join(args.outdir, "localization_summary.csv")
    write_csv(path, [row])
    print(f"Wrote localization summary to {path}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tracking = subparsers.add_parser("tracking", help="compute tracking metrics")
    tracking.add_argument("--manifest", default="")
    tracking.add_argument("--input", action="append", default=[])
    tracking.add_argument("--outdir", default="paper_results")
    tracking.add_argument("--controller", default="unknown")
    tracking.add_argument("--trajectory", default="unknown")
    tracking.add_argument("--run-id", default="")
    tracking.add_argument("--source", choices=["odom", "camera"], default="odom")
    tracking.add_argument("--start-time", type=float, default=0.0)
    tracking.add_argument("--end-time", type=float, default=None)
    tracking.add_argument("--epsilon", type=float, default=0.05)
    tracking.add_argument("--hold-time", type=float, default=2.0)
    tracking.add_argument("--no-plots", action="store_true")
    tracking.set_defaults(func=tracking_command)

    step = subparsers.add_parser("step", help="identify actuator step bandwidth")
    step.add_argument("--input", required=True)
    step.add_argument("--outdir", default="paper_results")
    step.add_argument("--kind", choices=["linear", "angular"], required=True)
    step.add_argument("--command-col", default="")
    step.add_argument("--response-col", default="")
    step.add_argument("--no-plots", action="store_true")
    step.set_defaults(func=step_command)

    link = subparsers.add_parser("link", help="compute ESP-NOW link metrics")
    link.add_argument("--input", required=True)
    link.add_argument("--outdir", default="paper_results")
    link.add_argument("--condition", default="")
    link.add_argument("--distance-m", type=float, default=float("nan"))
    link.add_argument("--nominal-period", type=float, default=0.05)
    link.set_defaults(func=link_command)

    localization = subparsers.add_parser(
        "localization", help="compare odometry against AprilTag/camera pose"
    )
    localization.add_argument("--input", required=True)
    localization.add_argument("--outdir", default="paper_results")
    localization.add_argument("--start-time", type=float, default=None)
    localization.add_argument("--end-time", type=float, default=None)
    localization.set_defaults(func=localization_command)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
