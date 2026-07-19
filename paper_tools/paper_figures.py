#!/usr/bin/env python3
"""Generate traceable publication figures from the frozen dataset registry."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_common import (
    camera_body_errors,
    load_registry,
    local_trajectory,
    read_csv_columns,
    save_figure,
    tracking_data,
)
from paper_style import COLORS, apply_style, finish_axis


def _start(ax, x, y):
    ax.plot(x, y, marker="*", markersize=7, color="black", linestyle="none", zorder=5)
    ax.annotate("Start", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)


def fig_circle(registry, output_dir):
    baseline = local_trajectory(tracking_data(registry["runs"]["circle_baseline"]))
    bsmc = local_trajectory(tracking_data(registry["runs"]["circle_bsmc"]))
    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    ax.plot(bsmc["desired_local_x"], bsmc["desired_local_y"], "--",
            color=COLORS["reference"], label="Reference", zorder=3)
    ax.plot(baseline["camera_local_x"], baseline["camera_local_y"],
            color=COLORS["baseline"], label="Backstepping", alpha=0.90)
    ax.plot(bsmc["camera_local_x"], bsmc["camera_local_y"],
            color=COLORS["bsmc"], label="BSMC", alpha=0.90)
    _start(ax, float(bsmc["desired_local_x"][0]), float(bsmc["desired_local_y"][0]))
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    finish_axis(ax, equal=True)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout(pad=0.3)
    paths = save_figure(fig, output_dir, "fig2_circle_xy", registry["paper"]["output_dpi"])
    plt.close(fig)
    return paths


def fig_eight(registry, output_dir):
    data = local_trajectory(tracking_data(registry["runs"]["eight_final"]))
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    ax.plot(data["desired_local_x"], data["desired_local_y"], "--",
            color=COLORS["reference"], label="Reference", zorder=3)
    ax.plot(data["camera_local_x"], data["camera_local_y"],
            color=COLORS["bsmc"], label="BSMC (camera)", alpha=0.82)
    _start(ax, float(data["desired_local_x"][0]), float(data["desired_local_y"][0]))
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    finish_axis(ax, equal=True)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout(pad=0.3)
    paths = save_figure(fig, output_dir, "fig3_eight_xy", registry["paper"]["output_dpi"])
    plt.close(fig)
    return paths


def fig_circle_errors(registry, output_dir):
    runs = [
        ("Backstepping", tracking_data(registry["runs"]["circle_baseline"])),
        ("BSMC", tracking_data(registry["runs"]["circle_bsmc"])),
    ]
    errors = [camera_body_errors(data) for _, data in runs]
    limits = []
    for row in range(3):
        peak = max(float(np.nanpercentile(np.abs(values[row]), 99.5)) for values in errors)
        limits.append(max(peak * 1.12, (0.01, 0.01, math.radians(1.0))[row]))

    fig, axes = plt.subplots(3, 2, figsize=(7.16, 4.65), sharex="col")
    labels = ("$e_x$ (m)", "$e_y$ (m)", "$e_\\theta$ (rad)")
    colors = (COLORS["camera"], COLORS["accent"], COLORS["bsmc"])
    for column, ((title, data), values) in enumerate(zip(runs, errors)):
        for row, series in enumerate(values):
            ax = axes[row, column]
            ax.plot(data["active_t"], series, color=colors[row], linewidth=0.9)
            ax.axhline(0.0, color="#666666", linewidth=0.6, linestyle="--")
            ax.set_ylim(-limits[row], limits[row])
            finish_axis(ax)
            if column == 0:
                ax.set_ylabel(labels[row])
            if row == 0:
                ax.text(0.02, 0.92, title, transform=ax.transAxes, va="top",
                        fontweight="bold", fontsize=8)
        axes[-1, column].set_xlabel("active time (s)")
    fig.tight_layout(w_pad=1.0, h_pad=0.45)
    paths = save_figure(fig, output_dir, "fig4_circle_errors", registry["paper"]["output_dpi"])
    plt.close(fig)
    return paths


def fig_square(registry, output_dir):
    selected = (
        ("square_1m_final", "(a) 1 m square"),
        ("square_2m_final", "(b) 2 m square"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.35))
    for ax, (name, panel) in zip(axes, selected):
        run = registry["runs"][name]
        data = local_trajectory(tracking_data(run))
        side = float(run["parameters"]["side_length_m"])
        ax.plot(data["desired_local_x"], data["desired_local_y"], "--",
                color=COLORS["reference"], label="Reference", zorder=3)
        ax.plot(data["camera_local_x"], data["camera_local_y"],
                color=COLORS["bsmc"], label="BSMC (camera)", alpha=0.90)
        vertices = np.asarray(((0, 0), (side, 0), (side, side), (0, side)))
        ax.scatter(vertices[:, 0], vertices[:, 1], marker="s", s=16,
                   facecolor="white", edgecolor="black", linewidth=0.7, zorder=5,
                   label="Vertices")
        _start(ax, 0.0, 0.0)
        ax.text(0.02, 0.98, panel, transform=ax.transAxes, va="top", fontweight="bold")
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        finish_axis(ax, equal=True)
    axes[0].legend(frameon=True, loc="best")
    fig.tight_layout(w_pad=1.0, pad=0.35)
    paths = save_figure(fig, output_dir, "fig5_square_xy", registry["paper"]["output_dpi"])
    plt.close(fig)
    return paths


def _estimated_gap_loss(interarrival, nominal_ms):
    received = len(interarrival)
    gaps = np.maximum(0, np.rint(interarrival / nominal_ms).astype(int) - 1)
    missing = int(gaps.sum())
    return missing, 100.0 * missing / max(1, received + missing)


def fig_link(registry, output_dir):
    nominal = float(registry["paper"]["nominal_period_ms"])
    values = []
    labels = []
    loss = []
    for run in registry["link_runs"]:
        columns = read_csv_columns(run["csv"])
        interarrival = columns["interarrival_ms"].astype(float)
        interarrival = interarrival[np.isfinite(interarrival) & (interarrival >= 0.0)]
        values.append(interarrival)
        labels.append(run["label"].replace(" ", "\n", 1))
        loss.append(_estimated_gap_loss(interarrival, nominal)[1])
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    box = ax.boxplot(values, showfliers=False, patch_artist=True, widths=0.62,
                     medianprops={"color": "black", "linewidth": 1.0})
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(COLORS["link"] if index % 2 == 0 else COLORS["accent"])
        patch.set_alpha(0.65)
    ax.axhline(nominal, color=COLORS["reference"], linestyle="--", linewidth=1.0,
               label="Nominal 50 ms")
    upper = max(float(np.percentile(item, 99.0)) for item in values)
    ax.set_ylim(0.0, upper * 1.24)
    for index, percentage in enumerate(loss, 1):
        ax.text(index, upper * 1.06, f"gap loss\n{percentage:.1f}%",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticklabels(labels)
    ax.set_ylabel("inter-arrival time (ms)")
    finish_axis(ax)
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout(pad=0.4)
    paths = save_figure(fig, output_dir, "fig6_espnow_interarrival", registry["paper"]["output_dpi"])
    plt.close(fig)
    return paths


FIGURES = {
    "circle": fig_circle,
    "eight": fig_eight,
    "errors": fig_circle_errors,
    "square": fig_square,
    "link": fig_link,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(Path(__file__).with_name("datasets.yaml")))
    parser.add_argument("--output-dir", default="paper_exports/figures")
    parser.add_argument("--only", choices=sorted(FIGURES), action="append")
    args = parser.parse_args()
    apply_style()
    registry = load_registry(args.registry)
    selected = args.only or list(FIGURES)
    for name in selected:
        paths = FIGURES[name](registry, args.output_dir)
        print(f"{name}: {paths[0]} | {paths[1]}")


if __name__ == "__main__":
    main()

