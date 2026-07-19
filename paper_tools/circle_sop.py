#!/usr/bin/env python3
"""Five-paired-run Circle SOP audit, statistics, figures, and tables.

The experimental unit is one independently restarted run.  Laps inside a run
are repeated measurements and are never counted as independent samples.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from paper_common import (
    camera_body_errors,
    local_trajectory,
    read_json,
    repo_path,
    save_figure,
    tracking_data,
)
from paper_style import COLORS, finish_axis


SHARED_PARAMETER_KEYS = (
    "radius_m", "angular_speed_rad_s", "k1", "k2", "k3", "phi1", "phi2",
    "max_v_m_s", "max_w_rad_s",
)


def configured(registry: dict) -> bool:
    section = registry.get("nominal_circle") or {}
    return bool(section.get("backstepping") or section.get("bsmc"))


def _entries(registry: dict, controller: str) -> list:
    return list((registry.get("nominal_circle") or {}).get(controller, []) or [])


def _csv_path(entry) -> Path:
    value = entry["csv"] if isinstance(entry, dict) else entry
    return repo_path(value)


def _summary_path(entry, csv_path: Path) -> Path:
    if isinstance(entry, dict) and entry.get("summary"):
        return repo_path(entry["summary"])
    return csv_path.with_name(csv_path.stem + "_summary.json")


def _run_label(entry, csv_path: Path) -> str:
    if isinstance(entry, dict) and entry.get("run_id"):
        return str(entry["run_id"])
    return csv_path.stem


def _lap_information(data: dict, tolerance_rad: float = 0.15):
    yaw = np.unwrap(np.asarray(data["desired_yaw"], dtype=float))
    direction = 1.0 if float(np.nanmedian(np.diff(yaw))) >= 0.0 else -1.0
    phase = direction * (yaw - yaw[0])
    span = float(np.nanmax(phase))
    ratio = span / (2.0 * math.pi)
    nearest = int(round(ratio))
    completed = nearest if abs(span - nearest * 2.0 * math.pi) <= tolerance_rad else int(math.floor(ratio))
    completed = max(0, completed)
    slices = []
    for lap in range(1, completed + 1):
        lo = (lap - 1) * 2.0 * math.pi
        hi = lap * 2.0 * math.pi
        indices = np.flatnonzero((phase >= lo - 1e-9) & (phase <= hi + 1e-9))
        if indices.size >= 20:
            slices.append(indices)
    boundaries = []
    for lap in range(1, completed):
        boundaries.append(int(np.argmin(np.abs(phase - lap * 2.0 * math.pi))))
    return phase, slices, boundaries, ratio


def _fresh_mask(data: dict, max_age_s: float):
    mask = np.ones(len(data["active_t"]), dtype=bool)
    if "camera_age_s" in data:
        age = np.asarray(data["camera_age_s"], dtype=float)
        mask &= np.isfinite(age) & (age <= max_age_s)
    for key in ("camera_x", "camera_y", "camera_yaw", "desired_x", "desired_y", "desired_yaw"):
        mask &= np.isfinite(np.asarray(data[key], dtype=float))
    return mask


def _convergence_time(t, error, valid, threshold_m, hold_s):
    start = None
    for index in range(len(t)):
        if valid[index] and error[index] <= threshold_m:
            if start is None:
                start = index
            if t[index] - t[start] >= hold_s:
                return float(t[start])
        else:
            start = None
    return math.nan


def load_run(entry, controller: str, config: dict) -> dict:
    csv_path = _csv_path(entry)
    run = {"csv": str(csv_path)}
    data = tracking_data(run, fresh_s=float(config.get("message_freshness_s", 0.30)))
    phase, lap_slices, boundaries, lap_ratio = _lap_information(data)
    ex, ey, etheta = camera_body_errors(data)
    position = np.hypot(np.asarray(data["desired_x"]) - np.asarray(data["camera_x"]),
                        np.asarray(data["desired_y"]) - np.asarray(data["camera_y"]))
    fresh = _fresh_mask(data, float(config.get("camera_max_age_s", 0.30)))
    ex = np.where(fresh, np.asarray(ex, dtype=float), np.nan)
    ey = np.where(fresh, np.asarray(ey, dtype=float), np.nan)
    etheta = np.where(fresh, np.asarray(etheta, dtype=float), np.nan)
    position = np.where(fresh, position, np.nan)
    metric = fresh & (np.asarray(data["active_t"]) >= float(config.get("transient_s", 5.0)))
    if int(metric.sum()) < 50:
        raise ValueError("fewer than 50 fresh post-transient camera samples")
    summary_path = _summary_path(entry, csv_path)
    summary = read_json(summary_path) if summary_path.exists() else {}
    parameters = dict(summary.get("parameters", {}))
    metrics = {
        "rmse_ex_m": float(np.sqrt(np.mean(ex[metric] ** 2))),
        "rmse_ey_m": float(np.sqrt(np.mean(ey[metric] ** 2))),
        "rmse_etheta_rad": float(np.sqrt(np.mean(etheta[metric] ** 2))),
        "rmse_position_m": float(np.sqrt(np.mean(position[metric] ** 2))),
        "max_position_m": float(np.max(position[metric])),
        "convergence_s": _convergence_time(
            np.asarray(data["active_t"]), position, fresh,
            float(config.get("convergence_threshold_m", 0.05)),
            float(config.get("convergence_hold_s", 1.0)),
        ),
    }
    per_lap = []
    for number, indices in enumerate(lap_slices, 1):
        use = indices[fresh[indices]]
        if use.size < 20:
            continue
        per_lap.append({
            "lap": number,
            "position_rmse_m": float(np.sqrt(np.mean(position[use] ** 2))),
            "heading_rmse_rad": float(np.sqrt(np.mean(etheta[use] ** 2))),
        })
    return {
        "controller": controller, "entry": entry, "csv": csv_path,
        "summary": summary_path, "run_id": _run_label(entry, csv_path),
        "data": data, "phase": phase, "lap_slices": lap_slices,
        "boundaries": boundaries, "lap_ratio": lap_ratio,
        "fresh": fresh, "errors": (ex, ey, etheta), "position": position,
        "parameters": parameters, "metrics": metrics, "per_lap": per_lap,
    }


def load_all(registry: dict) -> dict[str, list[dict]]:
    config = registry["nominal_circle"]
    return {
        controller: [load_run(entry, controller, config) for entry in _entries(registry, controller)]
        for controller in ("backstepping", "bsmc")
    }


def _different(left, right, tolerance=1e-9):
    try:
        return not math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left != right


def audit_circle_sop(registry: dict) -> tuple[list[str], list[str]]:
    if not configured(registry):
        return [], ["nominal_circle: repeat dataset not registered; Circle SOP assets will be skipped"]
    errors, warnings = [], []
    config = registry["nominal_circle"]
    required = int(config.get("required_runs_per_controller", 5))
    expected_laps = float(config.get("laps_per_run", 3))
    loaded = {}
    for controller in ("backstepping", "bsmc"):
        entries = _entries(registry, controller)
        if len(entries) != required:
            errors.append(f"nominal_circle.{controller}: expected {required} runs, found {len(entries)}")
            continue
        loaded[controller] = []
        for index, entry in enumerate(entries, 1):
            csv_path = _csv_path(entry)
            if not csv_path.exists():
                errors.append(f"nominal_circle.{controller}[{index}]: missing {csv_path}")
                continue
            summary_path = _summary_path(entry, csv_path)
            if not summary_path.exists():
                errors.append(
                    f"nominal_circle.{controller}[{index}]: missing companion summary "
                    f"{summary_path}"
                )
                continue
            try:
                item = load_run(entry, controller, config)
                loaded[controller].append(item)
            except (OSError, KeyError, ValueError) as exc:
                errors.append(f"nominal_circle.{controller}[{index}]: {exc}")
                continue
            if item["lap_ratio"] < expected_laps - 0.03:
                errors.append(
                    f"{item['run_id']}: only {item['lap_ratio']:.2f}/{expected_laps:g} requested laps completed"
                )
            values = np.asarray(item["data"].get("controller", []), dtype=object)
            observed = {str(value).strip().lower() for value in values if str(value).strip()}
            expected = "backstepping" if controller == "backstepping" else "bsmc"
            if not observed:
                errors.append(f"{item['run_id']}: controller column is empty")
            elif observed != {expected}:
                errors.append(f"{item['run_id']}: controller column={sorted(observed)}, expected {expected}")
            params = item["parameters"]
            if params:
                missing_parameters = [
                    key for key in (*SHARED_PARAMETER_KEYS, "ks1", "ks2")
                    if key not in params
                ]
                if missing_parameters:
                    errors.append(
                        f"{item['run_id']}: summary lacks parameters "
                        f"{', '.join(missing_parameters)}"
                    )
                ks1, ks2 = float(params.get("ks1", math.nan)), float(params.get("ks2", math.nan))
                if controller == "backstepping" and (abs(ks1) > 1e-12 or abs(ks2) > 1e-12):
                    errors.append(f"{item['run_id']}: Backstepping requires Ks1=Ks2=0")
                if controller == "bsmc" and not (ks1 > 0.0 and ks2 > 0.0):
                    errors.append(f"{item['run_id']}: BSMC requires Ks1>0 and Ks2>0")
            else:
                errors.append(
                    f"{item['run_id']}: summary has no parameters block; "
                    "cannot prove identical experimental settings"
                )
            fresh_fraction = float(np.mean(item["fresh"]))
            if fresh_fraction < 0.95:
                warnings.append(f"{item['run_id']}: fresh camera fraction only {100*fresh_fraction:.1f}%")
    if all(len(loaded.get(name, [])) == required for name in ("backstepping", "bsmc")):
        reference = loaded["backstepping"][0]["parameters"]
        for controller, runs in loaded.items():
            for item in runs:
                for key in SHARED_PARAMETER_KEYS:
                    if key in reference and key in item["parameters"] and _different(reference[key], item["parameters"][key]):
                        errors.append(
                            f"{item['run_id']}: {key}={item['parameters'][key]} differs from paired protocol value {reference[key]}"
                        )
        if len(set(item["run_id"] for runs in loaded.values() for item in runs)) != 2 * required:
            errors.append("nominal_circle: run IDs must be unique")
    representative = config.get("representative_block_index")
    if representative is not None:
        try:
            representative = int(representative)
        except (TypeError, ValueError):
            errors.append("nominal_circle.representative_block_index must be an integer")
        else:
            if not 1 <= representative <= required:
                errors.append(
                    "nominal_circle.representative_block_index must be between "
                    f"1 and {required}"
                )
    return errors, warnings


def _representative_index(groups: dict[str, list[dict]], config: dict) -> int:
    if "representative_block_index" in config:
        return int(config["representative_block_index"]) - 1
    bs = np.asarray([item["metrics"]["rmse_position_m"] for item in groups["backstepping"]])
    smc = np.asarray([item["metrics"]["rmse_position_m"] for item in groups["bsmc"]])
    score = np.abs(bs - np.median(bs)) / max(float(np.median(bs)), 1e-9)
    score += np.abs(smc - np.median(smc)) / max(float(np.median(smc)), 1e-9)
    return int(np.argmin(score))


def _last_lap_local(item):
    if not item["lap_slices"]:
        raise ValueError(f"{item['run_id']}: no complete lap")
    local = local_trajectory(item["data"])
    indices = item["lap_slices"][-1]
    return local, indices


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return math.nan, math.nan
    return float(np.mean(finite)), float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan


def _formatted(values, scale=1.0, digits=3):
    mean, std = _mean_std(values)
    return f"{scale*mean:.{digits}f} $\\pm$ {scale*std:.{digits}f}"


def _write_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]]):
    align = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[!t]", "\\centering", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}", f"\\begin{{tabular}}{{{align}}}", "\\hline",
        " & ".join(headers) + " \\\\", "\\hline",
    ]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_circle_sop_assets(registry: dict, output_root: str | Path):
    import matplotlib.pyplot as plt

    if not configured(registry):
        return []
    output_root = Path(output_root)
    figure_dir, table_dir = output_root / "figures", output_root / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    config = registry["nominal_circle"]
    groups = load_all(registry)
    rep_index = _representative_index(groups, config)
    representative = {name: runs[rep_index] for name, runs in groups.items()}

    # Figure 2: final complete lap of the same paired block for both methods.
    bs_local, bs_idx = _last_lap_local(representative["backstepping"])
    smc_local, smc_idx = _last_lap_local(representative["bsmc"])
    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    ax.plot(smc_local["desired_local_x"][smc_idx], smc_local["desired_local_y"][smc_idx],
            "--", color=COLORS["reference"], label="Reference", zorder=3)
    ax.plot(bs_local["camera_local_x"][bs_idx], bs_local["camera_local_y"][bs_idx],
            color=COLORS["baseline"], label="Backstepping (AprilTag)", alpha=0.90)
    ax.plot(smc_local["camera_local_x"][smc_idx], smc_local["camera_local_y"][smc_idx],
            color=COLORS["bsmc"], label="BSMC (AprilTag)", alpha=0.90)
    start = smc_idx[0]
    ax.plot(smc_local["desired_local_x"][start], smc_local["desired_local_y"][start],
            marker="*", markersize=7, color="black", linestyle="none", label="Start")
    ax.set_xlabel("$x$ (m)"); ax.set_ylabel("$y$ (m)")
    finish_axis(ax, equal=True); ax.legend(frameon=True, loc="best")
    fig.tight_layout(pad=0.3)
    outputs = list(save_figure(fig, figure_dir, "fig2_circle_xy", registry["paper"]["output_dpi"]))
    plt.close(fig)

    # Figure 4: all laps of the representative paired block; boundaries expose drift.
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 4.65), sharex="col")
    labels = ("$e_x$ (m)", "$e_y$ (m)", "$e_\\theta$ (rad)")
    colors = (COLORS["camera"], COLORS["accent"], COLORS["bsmc"])
    peaks = []
    for row in range(3):
        series = [np.abs(representative[name]["errors"][row][representative[name]["fresh"]])
                  for name in ("backstepping", "bsmc")]
        peaks.append(max(float(np.nanpercentile(value, 99.5)) for value in series) * 1.12)
    for column, name in enumerate(("backstepping", "bsmc")):
        item = representative[name]
        title = "Backstepping" if name == "backstepping" else "Compensated BSMC"
        for row, values in enumerate(item["errors"]):
            ax = axes[row, column]
            ax.plot(item["data"]["active_t"], values, color=colors[row], linewidth=0.85)
            ax.axhline(0.0, color="#666666", linewidth=0.6, linestyle="--")
            for boundary in item["boundaries"]:
                ax.axvline(item["data"]["active_t"][boundary], color="#777777",
                           linewidth=0.6, linestyle=":")
            ax.set_ylim(-max(peaks[row], 1e-3), max(peaks[row], 1e-3))
            finish_axis(ax)
            if column == 0: ax.set_ylabel(labels[row])
            if row == 0: ax.text(0.02, 0.92, title, transform=ax.transAxes,
                                 va="top", fontweight="bold", fontsize=8)
        axes[-1, column].set_xlabel("active time (s)")
    fig.tight_layout(w_pad=1.0, h_pad=0.45)
    outputs += list(save_figure(fig, figure_dir, "fig4_circle_errors", registry["paper"]["output_dpi"]))
    plt.close(fig)

    # Run-level statistics; n is independent starts, never lap count.
    metric_keys = ("rmse_ex_m", "rmse_ey_m", "rmse_etheta_rad", "max_position_m", "convergence_s")
    csv_rows, tex_rows = [], []
    for name, label in (("backstepping", "Backstepping"), ("bsmc", "Compensated BSMC")):
        values = {key: [item["metrics"][key] for item in groups[name]] for key in metric_keys}
        row = {"controller": label, "n": len(groups[name])}
        for key in metric_keys:
            mean, std = _mean_std(values[key]); row[key + "_mean"] = mean; row[key + "_std"] = std
        csv_rows.append(row)
        tex_rows.append([
            label, str(len(groups[name])), _formatted(values["rmse_ex_m"]),
            _formatted(values["rmse_ey_m"]), _formatted(values["rmse_etheta_rad"]),
            _formatted(values["max_position_m"]), _formatted(values["convergence_s"]),
        ])
    csv_path = table_dir / "table4_tracking_performance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    tex_path = table_dir / "table4_tracking_performance.tex"
    _write_table(
        tex_path,
        "Circle tracking performance across five independent hardware runs (mean $\\pm$ sample standard deviation).",
        "tab:tracking", ["Controller", "$n$", "RMSE $e_x$ (m)", "RMSE $e_y$ (m)",
                         "RMSE $e_\\theta$ (rad)", "Max. $e_p$ (m)", "$T_{conv}$ (s)"], tex_rows,
    )
    outputs += [csv_path, tex_path]

    # Parameters are read from the actual summaries and audited for parity.
    param_rows = []
    for name, label in (("backstepping", "Backstepping"), ("bsmc", "Compensated BSMC")):
        p = groups[name][0]["parameters"]
        param_rows.append([label] + [f"{float(p.get(key, math.nan)):.4g}" for key in
                          ("k1", "k2", "k3", "ks1", "ks2", "phi1", "phi2")])
    param_tex = table_dir / "table1_parameters.tex"
    _write_table(param_tex, "Controller parameters used in the Circle paired experiment.",
                 "tab:controller_parameters", ["Controller", "$k_1$", "$k_2$", "$k_3$",
                 "$K_{s1}$", "$K_{s2}$", "$\\phi_1$", "$\\phi_2$"], param_rows)
    outputs.append(param_tex)

    lap_csv = table_dir / "circle_per_lap_metrics.csv"
    lap_rows = []
    for name, runs in groups.items():
        for item in runs:
            for metric in item["per_lap"]:
                lap_rows.append({"controller": name, "run_id": item["run_id"], **metric})
    with lap_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(lap_rows[0])); writer.writeheader(); writer.writerows(lap_rows)
    outputs.append(lap_csv)

    provenance = {
        "experimental_unit": "independently restarted hardware run",
        "n_per_controller": len(groups["bsmc"]),
        "laps_per_run": config.get("laps_per_run", 3),
        "lap_independence": "laps are repeated measures, not independent n",
        "representative_rule": "paired block minimizing normalized distance to both controller medians",
        "representative_block_index_1_based": rep_index + 1,
        "representative_runs": {name: item["run_id"] for name, item in representative.items()},
        "registered_runs": {name: [item["run_id"] for item in runs] for name, runs in groups.items()},
        "figure_scope": "Fig. 2 last complete lap; Fig. 4 all laps with dotted lap boundaries",
        "processing": "raw AprilTag errors; rigid local-frame transform only; no smoothing",
    }
    provenance_path = output_root / "circle_sop_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    outputs.append(provenance_path)
    return outputs
