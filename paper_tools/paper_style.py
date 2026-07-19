"""Single-column, colorblind-safe matplotlib style for IEEE-like papers."""

from __future__ import annotations

import matplotlib as mpl


COLORS = {
    "reference": "#222222",
    "baseline": "#D55E00",
    "bsmc": "#0072B2",
    "camera": "#009E73",
    "accent": "#CC79A7",
    "link": "#56B4E9",
}


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.30,
        "figure.dpi": 120,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def finish_axis(ax, equal=False):
    ax.grid(True, alpha=0.30)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in", top=True, right=True)

