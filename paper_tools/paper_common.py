"""Shared data loading, validation, coordinate, and export helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
DEFAULT_REGISTRY = TOOLS_DIR / "datasets.yaml"


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        registry = yaml.safe_load(stream)
    registry["_registry_path"] = str(Path(path).resolve())
    return registry


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: str | Path) -> dict:
    with repo_path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_csv_columns(path: str | Path) -> dict[str, np.ndarray]:
    with repo_path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = []
        numeric = True
        for row in rows:
            try:
                values.append(float(row.get(key, "nan")))
            except (TypeError, ValueError):
                numeric = False
                break
        if numeric:
            columns[key] = np.asarray(values, dtype=float)
        else:
            columns[key] = np.asarray([row.get(key, "") for row in rows], dtype=object)
    return columns


TRACKING_KEYS = (
    "t", "ros_time", "odom_x", "odom_y", "odom_yaw", "camera_x",
    "camera_y", "camera_yaw", "desired_stamp", "desired_x", "desired_y",
    "desired_yaw", "cmd_stamp", "cmd_v", "cmd_w",
)


def tracking_data(run: dict, fresh_s: float = 0.30) -> dict[str, np.ndarray]:
    data = read_csv_columns(run["csv"])
    missing = [key for key in TRACKING_KEYS if key not in data]
    if missing:
        raise ValueError(f"{run['csv']} lacks columns: {', '.join(missing)}")
    valid = np.ones(len(data["t"]), dtype=bool)
    for key in TRACKING_KEYS:
        valid &= np.isfinite(data[key].astype(float))
    valid &= np.abs(data["ros_time"] - data["desired_stamp"]) <= fresh_s
    valid &= np.abs(data["ros_time"] - data["cmd_stamp"]) <= fresh_s
    if valid.sum() < 50:
        raise ValueError(f"not enough fresh samples: {run['csv']}")
    filtered = {key: value[valid] for key, value in data.items()}
    moving = np.flatnonzero(filtered["cmd_v"] > 0.01)
    if not len(moving):
        raise ValueError(f"no forward-motion samples: {run['csv']}")
    start = int(moving[0])
    filtered = {key: value[start:] for key, value in filtered.items()}
    filtered["active_t"] = filtered["t"] - filtered["t"][0]
    return filtered


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def to_local(x: np.ndarray, y: np.ndarray, x0: float, y0: float, yaw0: float):
    dx = x - x0
    dy = y - y0
    c = math.cos(yaw0)
    s = math.sin(yaw0)
    return c * dx + s * dy, -s * dx + c * dy


def local_trajectory(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x0 = float(data["desired_x"][0])
    y0 = float(data["desired_y"][0])
    yaw0 = float(data["desired_yaw"][0])
    desired_x, desired_y = to_local(data["desired_x"], data["desired_y"], x0, y0, yaw0)
    camera_x, camera_y = to_local(data["camera_x"], data["camera_y"], x0, y0, yaw0)
    odom_x, odom_y = to_local(data["odom_x"], data["odom_y"], x0, y0, yaw0)
    return {
        **data,
        "desired_local_x": desired_x,
        "desired_local_y": desired_y,
        "camera_local_x": camera_x,
        "camera_local_y": camera_y,
        "odom_local_x": odom_x,
        "odom_local_y": odom_y,
    }


def camera_body_errors(data: dict[str, np.ndarray]):
    dx = data["desired_x"] - data["camera_x"]
    dy = data["desired_y"] - data["camera_y"]
    yaw = data["camera_yaw"]
    ex = np.cos(yaw) * dx + np.sin(yaw) * dy
    ey = -np.sin(yaw) * dx + np.cos(yaw) * dy
    etheta = wrap_angle(data["desired_yaw"] - yaw)
    return ex, ey, etheta


def rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(np.square(finite)))) if len(finite) else math.nan


def save_figure(fig, output_dir: str | Path, stem: str, dpi: int = 600):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    return pdf, png

