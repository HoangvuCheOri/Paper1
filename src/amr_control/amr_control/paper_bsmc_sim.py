#!/usr/bin/env python3
"""Offline BSMC vs backstepping simulation for paper ablation studies."""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Gains:
    k1: float = 0.8
    k2: float = 2.4
    k3: float = 4.0
    c: float = 1.0
    ks1: float = 0.002
    ks2: float = 0.005
    phi1: float = 0.8
    phi2: float = 1.2
    max_v: float = 0.35
    max_w: float = 0.6


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def sat(value):
    return max(-1.0, min(1.0, value))


def desired_circle(t, radius, omega, ramp_time):
    vd_nominal = radius * omega
    if t < ramp_time:
        s = vd_nominal * t * t / (2.0 * ramp_time)
        vd = vd_nominal * t / ramp_time
        wd = omega * t / ramp_time
    else:
        s = vd_nominal * (t - ramp_time / 2.0)
        vd = vd_nominal
        wd = omega
    angle = s / radius
    x = radius * math.sin(angle)
    y = radius * (1.0 - math.cos(angle))
    theta = wrap_angle(angle)
    return x, y, theta, vd, wd


def desired_eight(t, amplitude, omega, ramp_time):
    ramp = min(1.0, max(0.0, t / ramp_time))
    wt = omega * t
    x = amplitude * math.sin(wt)
    y = 0.5 * amplitude * math.sin(2.0 * wt)
    dx = amplitude * omega * math.cos(wt) * ramp
    dy = amplitude * omega * math.cos(2.0 * wt) * ramp
    ddx = -amplitude * omega * omega * math.sin(wt) * ramp
    ddy = -2.0 * amplitude * omega * omega * math.sin(2.0 * wt) * ramp
    vd = math.hypot(dx, dy)
    theta = math.atan2(dy, dx) if vd > 1e-12 else 0.0
    wd = (dx * ddy - dy * ddx) / (vd * vd) if vd > 1e-12 else 0.0
    return x, y, wrap_angle(theta), vd, wd


def desired_pose(t, trajectory, radius, amplitude, omega, ramp_time):
    if trajectory == "circle":
        return desired_circle(t, radius, omega, ramp_time)
    return desired_eight(t, amplitude, omega, ramp_time)


def controller(state, desired, gains):
    x, y, theta, _, _ = state
    xd, yd, thetad, vd, wd = desired

    dx = xd - x
    dy = yd - y
    ex = math.cos(theta) * dx + math.sin(theta) * dy
    ey = -math.sin(theta) * dx + math.cos(theta) * dy
    etheta = wrap_angle(thetad - theta)

    s1 = ex
    s2 = etheta + gains.c * ey
    vcmd = vd * math.cos(etheta) + gains.k1 * ex + gains.ks1 * sat(s1 / gains.phi1)
    wcmd = (
        wd
        + vd * (gains.k2 * ey + gains.k3 * math.sin(etheta))
        + gains.ks2 * sat(s2 / gains.phi2)
    )

    vcmd = max(0.0, min(gains.max_v, vcmd))
    wcmd = max(-gains.max_w, min(gains.max_w, wcmd))
    return vcmd, wcmd, ex, ey, etheta


def simulate_run(args, controller_name, gains, rng):
    desired0 = desired_pose(
        0.0, args.trajectory, args.radius, args.amplitude, args.omega, args.ramp_time
    )
    x = desired0[0] - args.initial_x_error
    y = desired0[1] - args.initial_y_error
    theta = wrap_angle(desired0[2] - math.radians(args.initial_heading_error_deg))
    v = 0.0
    w = 0.0

    rows = []
    n_steps = int(args.duration / args.dt) + 1
    for step in range(n_steps):
        t = step * args.dt
        desired = desired_pose(
            t,
            args.trajectory,
            args.radius,
            args.amplitude,
            args.omega,
            args.ramp_time,
        )
        vcmd, wcmd, ex, ey, etheta = controller((x, y, theta, v, w), desired, gains)

        dist_v = args.linear_bias + args.disturbance_amp * math.sin(2.0 * math.pi * 0.20 * t)
        dist_w = args.angular_bias + args.disturbance_amp * math.sin(2.0 * math.pi * 0.17 * t)
        noise_v = rng.normal(0.0, args.velocity_noise)
        noise_w = rng.normal(0.0, args.angular_noise)

        v += args.dt * (args.kv * (vcmd - v) + dist_v + noise_v)
        w += args.dt * (args.kw * (wcmd - w) + dist_w + noise_w)
        x += v * math.cos(theta) * args.dt
        y += v * math.sin(theta) * args.dt
        theta = wrap_angle(theta + w * args.dt)

        rows.append(
            {
                "t": t,
                "controller": controller_name,
                "trajectory": args.trajectory,
                "run_id": args.run_id,
                "odom_x": x,
                "odom_y": y,
                "odom_yaw": theta,
                "odom_v": v,
                "odom_w": w,
                "desired_x": desired[0],
                "desired_y": desired[1],
                "desired_yaw": desired[2],
                "error_ex": ex,
                "error_ey": ey,
                "error_etheta": etheta,
                "cmd_v": vcmd,
                "cmd_w": wcmd,
            }
        )
    return rows


def write_rows(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    ex = np.asarray([row["error_ex"] for row in rows], dtype=float)
    ey = np.asarray([row["error_ey"] for row in rows], dtype=float)
    etheta = np.asarray([row["error_etheta"] for row in rows], dtype=float)
    ep = np.hypot(
        np.asarray([row["desired_x"] - row["odom_x"] for row in rows], dtype=float),
        np.asarray([row["desired_y"] - row["odom_y"] for row in rows], dtype=float),
    )
    return {
        "controller": rows[0]["controller"],
        "trajectory": rows[0]["trajectory"],
        "run_id": rows[0]["run_id"],
        "duration_s": rows[-1]["t"] - rows[0]["t"],
        "rmse_ex_m": float(np.sqrt(np.mean(ex * ex))),
        "rmse_ey_m": float(np.sqrt(np.mean(ey * ey))),
        "rmse_etheta_rad": float(np.sqrt(np.mean(etheta * etheta))),
        "rmse_position_m": float(np.sqrt(np.mean(ep * ep))),
        "mae_position_m": float(np.mean(np.abs(ep))),
        "max_position_m": float(np.max(ep)),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", choices=["circle", "eight"], default="circle")
    parser.add_argument("--outdir", default="paper_results/sim")
    parser.add_argument("--run-id", default="sim")
    parser.add_argument("--duration", type=float, default=80.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--omega", type=float, default=0.20)
    parser.add_argument("--ramp-time", type=float, default=2.5)
    parser.add_argument("--kv", type=float, default=8.0)
    parser.add_argument("--kw", type=float, default=8.0)
    parser.add_argument("--linear-bias", type=float, default=0.0)
    parser.add_argument("--angular-bias", type=float, default=0.0)
    parser.add_argument("--disturbance-amp", type=float, default=0.01)
    parser.add_argument("--velocity-noise", type=float, default=0.002)
    parser.add_argument("--angular-noise", type=float, default=0.004)
    parser.add_argument("--initial-x-error", type=float, default=0.05)
    parser.add_argument("--initial-y-error", type=float, default=-0.03)
    parser.add_argument("--initial-heading-error-deg", type=float, default=5.0)
    parser.add_argument("--k1", type=float, default=0.8)
    parser.add_argument("--k2", type=float, default=2.4)
    parser.add_argument("--k3", type=float, default=4.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--ks1", type=float, default=0.002)
    parser.add_argument("--ks2", type=float, default=0.005)
    parser.add_argument("--phi1", type=float, default=0.8)
    parser.add_argument("--phi2", type=float, default=1.2)
    parser.add_argument("--max-v", type=float, default=0.35)
    parser.add_argument("--max-w", type=float, default=0.6)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    base_gains = Gains(
        k1=args.k1,
        k2=args.k2,
        k3=args.k3,
        c=args.c,
        ks1=args.ks1,
        ks2=args.ks2,
        phi1=args.phi1,
        phi2=args.phi2,
        max_v=args.max_v,
        max_w=args.max_w,
    )
    baseline_gains = Gains(
        k1=args.k1,
        k2=args.k2,
        k3=args.k3,
        c=args.c,
        ks1=0.0,
        ks2=0.0,
        phi1=args.phi1,
        phi2=args.phi2,
        max_v=args.max_v,
        max_w=args.max_w,
    )

    summary = []
    manifest_rows = []
    for controller_name, gains, seed_offset in [
        ("Backstepping", baseline_gains, 0),
        ("BSMC", base_gains, 1000),
    ]:
        rng = np.random.default_rng(args.seed + seed_offset)
        rows = simulate_run(args, controller_name, gains, rng)
        file_name = f"{args.trajectory}_{controller_name.lower()}_{args.run_id}.csv"
        path = os.path.join(args.outdir, file_name)
        write_rows(path, rows)
        summary.append(summarize(rows))
        manifest_rows.append(
            {
                "file": file_name,
                "controller": controller_name,
                "trajectory": args.trajectory,
                "run_id": args.run_id,
                "source": "odom",
                "start_time": 0.0,
                "end_time": "",
            }
        )

    write_rows(os.path.join(args.outdir, "simulation_summary.csv"), summary)
    write_rows(os.path.join(args.outdir, "manifest.csv"), manifest_rows)
    print(f"Wrote simulation CSV, summary, and manifest to {args.outdir}")
    print(
        "Run paper_metrics tracking --manifest "
        f"{os.path.join(args.outdir, 'manifest.csv')} --outdir {args.outdir}"
    )


if __name__ == "__main__":
    main()
