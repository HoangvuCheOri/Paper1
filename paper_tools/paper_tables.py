#!/usr/bin/env python3
"""Generate CSV and LaTeX tables from registered paper datasets."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from paper_common import load_registry, read_csv_columns, read_json, repo_path


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value):
    return str(value).replace("_", "\\_").replace("%", "\\%")


def _write_latex(path: Path, rows: list[dict], columns: list[tuple[str, str]], caption: str, label: str):
    align = "l" + "r" * (len(columns) - 1)
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{" + align + "}",
        "\\hline",
        " & ".join(header for _, header in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row[key]) for key, _ in columns) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _gap_metrics(path, nominal_ms):
    columns = read_csv_columns(path)
    dt = columns["interarrival_ms"].astype(float)
    dt = dt[np.isfinite(dt) & (dt >= 0.0)]
    gaps = np.maximum(0, np.rint(dt / nominal_ms).astype(int) - 1)
    missing = int(gaps.sum())
    loss = 100.0 * missing / max(1, len(dt) + missing)
    return {
        "packets": len(dt),
        "median_ms": float(np.median(dt)),
        "p95_ms": float(np.percentile(dt, 95.0)),
        "p99_ms": float(np.percentile(dt, 99.0)),
        "gap_loss_percent": loss,
    }


def protocol_rows(registry):
    rows = []
    for run in registry["runs"].values():
        data = read_csv_columns(run["csv"])
        rows.append({
            "trajectory": run["trajectory"],
            "method": run["label"],
            "run_id": run["run_id"],
            "n": 1,
            "samples": len(data["t"]),
            "duration_s": f"{float(np.nanmax(data['t']) - np.nanmin(data['t'])):.1f}",
            "feedback": "camera--wheel EKF",
            "primary_metric": "raw AprilTag",
        })
    return rows


def performance_rows(registry):
    rows = []
    for key in ("circle_baseline", "circle_bsmc", "eight_final", "square_1m_final", "square_2m_final"):
        run = registry["runs"][key]
        summary = read_json(run["summary"])
        if run["trajectory"] == "square":
            position = summary["overall_position_rmse_m"]
            path = summary["straight_lateral_rmse_m"]
            heading = summary["straight_heading_rmse_deg"]
            note = "heading/path on established straight edges"
        else:
            position = summary["position_rmse_m"]
            path = summary["path_rmse_m"]
            heading = summary["heading_rmse_deg"]
            note = "steady-state after 5 s transient"
        rows.append({
            "trajectory": run["trajectory"],
            "method": run["label"],
            "n": 1,
            "position_rmse_cm": f"{100.0 * position:.2f}",
            "path_rmse_cm": f"{100.0 * path:.2f}",
            "heading_rmse_deg": f"{heading:.2f}",
            "scope": note,
        })
    return rows


def controller_rows(registry):
    rows = []
    for key in ("circle_bsmc", "eight_final", "square_1m_final", "square_2m_final"):
        run = registry["runs"][key]
        p = run["parameters"]
        profile_name = {
            "circle_bsmc": "Circle BSMC",
            "eight_final": "Figure-eight BSMC",
            "square_1m_final": "Square 1 m BSMC",
            "square_2m_final": "Square 2 m BSMC",
        }[key]
        rows.append({
            "profile": profile_name,
            "k1": f"{p['k1']:.3g}", "k2": f"{p['k2']:.3g}", "k3": f"{p['k3']:.3g}",
            "ks1": f"{p['ks1']:.3g}", "ks2": f"{p['ks2']:.3g}",
            "phi1": f"{p['phi1']:.3g}", "phi2": f"{p['phi2']:.3g}",
        })
    return rows


def link_rows(registry):
    nominal = float(registry["paper"]["nominal_period_ms"])
    rows = []
    for run in registry["link_runs"]:
        metric = _gap_metrics(run["csv"], nominal)
        rows.append({
            "condition": run["label"],
            "packets": metric["packets"],
            "median_ms": f"{metric['median_ms']:.2f}",
            "p95_ms": f"{metric['p95_ms']:.2f}",
            "p99_ms": f"{metric['p99_ms']:.2f}",
            "gap_loss_percent": f"{metric['gap_loss_percent']:.2f}",
            "latency": "N/A",
        })
    return rows


def build(registry, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specs = [
        ("table_protocol", protocol_rows(registry),
         [("trajectory", "Trajectory"), ("method", "Method"), ("run_id", "Run ID"),
          ("n", "$n$"), ("duration_s", "Duration (s)")],
         "Registered experimental runs used by the main figures.", "tab:protocol"),
        ("table_tracking", performance_rows(registry),
         [("trajectory", "Trajectory"), ("method", "Method"), ("n", "$n$"),
          ("position_rmse_cm", "Position (cm)"), ("path_rmse_cm", "Path (cm)"),
          ("heading_rmse_deg", "Heading (deg)")],
         "Tracking performance from raw AprilTag measurements. Values are single-run results. "
         "Circle and figure-eight metrics exclude the first 5 s of motion; square position RMSE "
         "uses the complete moving interval, whereas square path and heading RMSE use established "
         "straight-edge samples only.", "tab:tracking"),
        ("table_controller_profiles", controller_rows(registry),
         [("profile", "Profile"), ("k1", "$k_1$"), ("k2", "$k_2$"), ("k3", "$k_3$"),
          ("ks1", "$K_{s1}$"), ("ks2", "$K_{s2}$"), ("phi1", "$\\phi_1$"), ("phi2", "$\\phi_2$")],
         "Final trajectory-specific controller profiles.", "tab:controller_profiles"),
        ("table_espnow", link_rows(registry),
         [("condition", "Condition"), ("packets", "Packets"), ("median_ms", "Median (ms)"),
          ("p95_ms", "P95 (ms)"), ("p99_ms", "P99 (ms)"),
          ("gap_loss_percent", "Gap loss (\\%)")],
         "ESP-NOW inter-arrival statistics. Loss is estimated from timing gaps because packet sequence numbers were unavailable.",
         "tab:espnow"),
    ]
    for stem, rows, columns, caption, label in specs:
        _write_csv(output / f"{stem}.csv", rows)
        _write_latex(output / f"{stem}.tex", rows, columns, caption, label)
        print(f"{stem}: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(Path(__file__).with_name("datasets.yaml")))
    parser.add_argument("--output-dir", default="paper_exports/tables")
    args = parser.parse_args()
    build(load_registry(args.registry), args.output_dir)


if __name__ == "__main__":
    main()
