#!/usr/bin/env python3
"""Audit publication datasets without modifying them."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from circle_sop import audit_circle_sop
from paper_common import load_registry, read_csv_columns, repo_path, tracking_data, wrap_angle


def audit(registry: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required = {"t", "ros_time", "camera_x", "camera_y", "camera_yaw",
                "desired_x", "desired_y", "desired_yaw", "cmd_v", "cmd_w"}
    roles: dict[str, set[str]] = {}
    for name, run in registry["runs"].items():
        csv_path = repo_path(run["csv"])
        summary_path = repo_path(run["summary"])
        if not csv_path.exists():
            errors.append(f"{name}: missing CSV {csv_path}")
            continue
        if not summary_path.exists():
            errors.append(f"{name}: missing summary {summary_path}")
        columns = read_csv_columns(csv_path)
        missing = sorted(required - set(columns))
        if missing:
            errors.append(f"{name}: missing columns {', '.join(missing)}")
            continue
        try:
            data = tracking_data(run)
            max_age = max(
                float(np.max(np.abs(data["ros_time"] - data["desired_stamp"]))),
                float(np.max(np.abs(data["ros_time"] - data["cmd_stamp"]))),
            )
            if max_age > 0.30:
                warnings.append(f"{name}: cached desired/command samples exceed 0.30 s")
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
        roles.setdefault(run["trajectory"], set()).add(run["role"])

        if run["trajectory"] == "square":
            desired = columns["desired_yaw"].astype(float)
            jumps = wrap_angle(np.diff(desired))
            corner = np.abs(jumps) > math.radians(60.0)
            if corner.any():
                median = float(np.median(np.abs(jumps[corner])))
                warnings.append(
                    f"{name}: desired heading has {corner.sum()} discrete corner jumps; "
                    f"median={math.degrees(median):.1f} deg (expected exact-polyline behavior)"
                )

    for trajectory in ("circle", "eight", "square"):
        available = roles.get(trajectory, set())
        if not {"baseline", "compensated"}.issubset(available):
            warnings.append(
                f"{trajectory}: no complete baseline/compensated pair in registry; "
                "do not claim a paired controller comparison"
            )

    for link in registry["link_runs"]:
        path = repo_path(link["csv"])
        if not path.exists():
            errors.append(f"link {link['label']}: missing {path}")
            continue
        columns = read_csv_columns(path)
        seq = columns.get("seq", np.asarray([])).astype(float)
        robot_ms = columns.get("robot_time_ms", np.asarray([])).astype(float)
        if not np.isfinite(seq).any():
            warnings.append(f"link {link['label']}: no sequence numbers; loss is gap-estimated only")
        if not np.isfinite(robot_ms).any():
            warnings.append(f"link {link['label']}: no robot timestamps; one-way latency unavailable")
    circle_errors, circle_warnings = audit_circle_sop(registry)
    errors.extend(circle_errors)
    warnings.extend(circle_warnings)
    return errors, warnings


def render(errors: list[str], warnings: list[str]) -> str:
    lines = ["# Publication data audit", ""]
    lines += [f"- ERROR: {item}" for item in errors] or ["- No blocking errors."]
    lines += ["", "## Scientific warnings", ""]
    lines += [f"- {item}" for item in warnings] or ["- None."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(Path(__file__).with_name("datasets.yaml")))
    parser.add_argument("--output", default="", help="optional Markdown report path")
    args = parser.parse_args()
    errors, warnings = audit(load_registry(args.registry))
    report = render(errors, warnings)
    print(report, end="")
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
