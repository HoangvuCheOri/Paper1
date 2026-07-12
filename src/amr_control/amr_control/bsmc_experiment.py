#!/usr/bin/env python3
"""Preflight, run, and rank camera-EKF BSMC experiments without launch files."""

import argparse
import csv
import math
import os
import random
import signal
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def finite(value):
    return math.isfinite(value)


def rmse(values):
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def angle_std(values):
    if len(values) <= 1:
        return 0.0
    reference = values[0]
    return sample_std([wrap_angle(value - reference) for value in values])


def angle_mean(values):
    if not values:
        return math.nan
    return math.atan2(
        sum(math.sin(value) for value in values),
        sum(math.cos(value) for value in values),
    )


class SensorCheck(Node):
    def __init__(self):
        super().__init__("bsmc_sensor_check")
        self.samples = {"camera": [], "filtered": [], "raw": []}
        self.create_subscription(Odometry, "/odom_camera", self.camera_cb, 20)
        self.create_subscription(Odometry, "/odometry/filtered", self.filtered_cb, 20)
        self.create_subscription(Odometry, "/odom_raw", self.raw_cb, 20)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def append(self, name, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.samples[name].append(
            (
                self.now(),
                stamp,
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                yaw_from_quaternion(msg.pose.pose.orientation),
            )
        )

    def camera_cb(self, msg):
        self.append("camera", msg)

    def filtered_cb(self, msg):
        self.append("filtered", msg)

    def raw_cb(self, msg):
        self.append("raw", msg)


def sensor_summary(samples, duration):
    output = {}
    for name, rows in samples.items():
        rate = len(rows) / max(duration, 1e-6)
        ages = [received - stamp for received, stamp, *_ in rows if stamp > 0.0]
        output[name] = {
            "count": len(rows),
            "rate_hz": rate,
            "age_mean_s": statistics.mean(ages) if ages else math.nan,
            "age_max_s": max(ages) if ages else math.nan,
            "std_x_m": sample_std([row[2] for row in rows]),
            "std_y_m": sample_std([row[3] for row in rows]),
            "std_yaw_rad": angle_std([row[4] for row in rows]),
            "mean_x_m": statistics.mean([row[2] for row in rows]) if rows else math.nan,
            "mean_y_m": statistics.mean([row[3] for row in rows]) if rows else math.nan,
            "mean_yaw_rad": angle_mean([row[4] for row in rows]),
        }
    return output


def collect_sensor_check(duration):
    rclpy.init(args=None)
    node = SensorCheck()
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        samples = node.samples
        node.destroy_node()
        rclpy.shutdown()
    return sensor_summary(samples, time.monotonic() - started)


def print_sensor_summary(summary):
    print("sensor,count,rate_hz,mean_age_ms,max_age_ms,std_x_cm,std_y_cm,std_yaw_deg")
    for name in ("camera", "filtered", "raw"):
        row = summary[name]
        print(
            f"{name},{row['count']},{row['rate_hz']:.2f},"
            f"{1000.0 * row['age_mean_s']:.1f},{1000.0 * row['age_max_s']:.1f},"
            f"{100.0 * row['std_x_m']:.3f},{100.0 * row['std_y_m']:.3f},"
            f"{math.degrees(row['std_yaw_rad']):.3f}"
        )


def preflight_ok(summary):
    camera = summary["camera"]
    filtered = summary["filtered"]
    raw = summary["raw"]
    problems = []
    if camera["rate_hz"] < 5.0:
        problems.append(f"/odom_camera rate too low ({camera['rate_hz']:.1f} Hz)")
    if filtered["rate_hz"] < 10.0:
        problems.append(f"/odometry/filtered rate too low ({filtered['rate_hz']:.1f} Hz)")
    if raw["rate_hz"] < 10.0:
        problems.append(f"/odom_raw rate too low ({raw['rate_hz']:.1f} Hz)")
    if not finite(camera["age_max_s"]):
        problems.append("camera messages have invalid/zero timestamps")
    elif camera["age_max_s"] > 0.30:
        problems.append(f"camera timestamp age too high ({camera['age_max_s']:.3f} s)")
    if camera["std_x_m"] > 0.03 or camera["std_y_m"] > 0.03:
        problems.append("camera static XY noise exceeds 3 cm (or robot is moving)")
    if camera["std_yaw_rad"] > math.radians(2.0):
        problems.append("camera static yaw noise exceeds 2 deg (or robot is moving)")
    if filtered["std_x_m"] > 0.03 or filtered["std_y_m"] > 0.03:
        problems.append("filtered odometry static XY variation exceeds 3 cm")
    if filtered["std_yaw_rad"] > math.radians(2.0):
        problems.append("filtered odometry static yaw variation exceeds 2 deg")
    return problems


DEFAULT_GAINS = {
    "circle": {"k1": 0.40, "k2": 2.4, "k3": 3.5, "ks1": 0.20, "ks2": 0.40, "phi1": 1.0, "phi2": 1.5},
    "eight": {"k1": 1.00, "k2": 3.0, "k3": 3.5, "ks1": 0.08, "ks2": 0.10, "phi1": 0.4, "phi2": 1.0},
    # bsmc_square defaults to Backstepping; these conservative SMC gains
    # provide the BSMC arm of the controlled comparison.
    "square": {"k1": 0.80, "k2": 6.0, "k3": 6.0, "ks1": 0.08, "ks2": 0.10, "phi1": 0.5, "phi2": 0.8},
}

DEFAULT_DURATION = {"circle": 63.0, "eight": 45.0, "square": 44.0}


def append_manifest(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(row.keys())
    if exists:
        with path.open(newline="") as stream:
            old_rows = list(csv.DictReader(stream))
        old_fields = list(old_rows[0].keys()) if old_rows else []
        if old_fields != fields:
            fields = old_fields + [field for field in fields if field not in old_fields]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(old_rows)
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def terminate_process(process, timeout=2.0):
    """Stop a subprocess; escalate SIGINT → SIGTERM → SIGKILL on process group.

    Blocks SIGINT during cleanup so repeated Ctrl-C cannot leave orphans.
    """
    if process is None or process.poll() is not None:
        return
    # Block SIGINT during cleanup to prevent orphaned processes
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        # 1. Graceful: SIGINT
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        # 2. Escalate: SIGTERM to entire process group
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except OSError:
            process.terminate()
        try:
            process.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        # 3. Last resort: SIGKILL the entire process group
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except OSError:
            process.kill()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    finally:
        signal.signal(signal.SIGINT, old_handler)


def validate_run_args(args):
    if args.duration < 0.0:
        raise SystemExit("--duration must be >= 0")
    if args.check_duration < 2.0:
        raise SystemExit("--check-duration must be >= 2 s")
    # Only validate trajectory-specific geometry params
    if args.trajectory == "circle":
        if args.radius <= 0.0:
            raise SystemExit("--radius must be positive")
    elif args.trajectory == "eight":
        if args.amplitude <= 0.0:
            raise SystemExit("--amplitude must be positive")
    elif args.trajectory == "square":
        if args.side_length <= 0.0:
            raise SystemExit("--side-length must be positive")
        if not 0.01 <= args.corner_radius <= args.side_length / 2.0:
            raise SystemExit("--corner-radius must be between 0.01 and side_length/2")
    defaults = DEFAULT_GAINS[args.trajectory]
    gains = {
        key: defaults[key] if getattr(args, key) is None else getattr(args, key)
        for key in defaults
    }
    for key in ("k1", "k2", "k3", "ks1", "ks2"):
        if gains[key] < 0.0:
            raise SystemExit(f"--{key} must be >= 0")
    for key in ("phi1", "phi2"):
        if gains[key] <= 0.0:
            raise SystemExit(f"--{key} must be > 0")
    for key in ("yaw_bias_gain", "radius_feedback_gain", "radius_position_gain"):
        if getattr(args, key) < 0.0:
            raise SystemExit(f"--{key.replace('_', '-')} must be >= 0")


def run_experiment(args):
    validate_run_args(args)
    # Preflight with retry — robot may still be decelerating from previous run
    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"Checking camera and EKF for {args.check_duration:.1f} s...")
        summary = collect_sensor_check(args.check_duration)
        print_sensor_summary(summary)
        problems = preflight_ok(summary)
        if not problems:
            break
        if attempt + 1 < max_attempts:
            wait = 5.0 * (attempt + 1)
            print(
                f"Preflight issue (attempt {attempt + 1}/{max_attempts}): "
                f"{'; '.join(problems)}"
            )
            print(f"Robot may still be moving. Waiting {wait:.0f}s before retry...")
            time.sleep(wait)
        elif not args.force:
            raise SystemExit("Preflight failed after retries: " + "; ".join(problems))
        else:
            print("WARNING: " + "; ".join(problems))

    run_summary = summary
    if not args.no_reset:
        print("Resetting raw odometry and EKF before the run...")
        subprocess.run(
            ["ros2", "topic", "pub", "--once", "/reset_odom", "std_msgs/msg/Empty", "{}"],
            check=True,
            timeout=8.0,
        )
        time.sleep(2.0)
        post_reset = collect_sensor_check(2.0)
        run_summary = post_reset
        post_reset_problems = preflight_ok(post_reset)
        if post_reset_problems and not args.force:
            raise SystemExit("Post-reset check failed: " + "; ".join(post_reset_problems))

    start_pose = (
        run_summary["camera"]["mean_x_m"],
        run_summary["camera"]["mean_y_m"],
        run_summary["camera"]["mean_yaw_rad"],
    )
    start_reference = getattr(args, "start_reference", None)
    if start_reference is not None:
        position_error = math.hypot(
            start_pose[0] - start_reference[0], start_pose[1] - start_reference[1]
        )
        yaw_error = abs(wrap_angle(start_pose[2] - start_reference[2]))
        if (
            position_error > args.start_position_tolerance
            or yaw_error > math.radians(args.start_yaw_tolerance_deg)
        ):
            print(
                f"WARNING: Start pose moved: {position_error:.3f} m, "
                f"{math.degrees(yaw_error):.1f} deg. Continuing anyway..."
            )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    defaults = DEFAULT_GAINS[args.trajectory]
    gains = {
        key: defaults[key] if getattr(args, key) is None else getattr(args, key)
        for key in defaults
    }
    if args.controller == "backstepping":
        gains["ks1"] = 0.0
        gains["ks2"] = 0.0
    duration = args.duration if args.duration > 0.0 else DEFAULT_DURATION[args.trajectory]
    odom_topic = "/odometry/filtered" if args.source == "fusion" else "/odom_raw"
    run_id = args.run_id or f"{args.trajectory}_{args.controller}_{args.source}_{stamp}"
    controller_name = "BSMC" if args.controller == "bsmc" else "Backstepping"
    csv_path = output_dir / f"{stamp}_{args.trajectory}_{controller_name}_{args.source}_{run_id}.csv"

    logger_cmd = [
        "ros2", "run", "amr_control", "paper_data_logger", "--ros-args",
        "-p", f"output_file:={csv_path}",
        "-p", f"controller:={controller_name}",
        "-p", f"trajectory:={args.trajectory}",
        "-p", f"run_id:={run_id}",
        "-p", f"odom_topic:={odom_topic}",
    ]
    executable = {"circle": "bsmc_circle", "eight": "bsmc_eight", "square": "bsmc_square"}[args.trajectory]
    controller_cmd = [
        "ros2", "run", "amr_control", executable, "--ros-args",
        "-p", f"odom_topic:={odom_topic}",
        "-p", f"k1:={gains['k1']}", "-p", f"k2:={gains['k2']}",
        "-p", f"k3:={gains['k3']}", "-p", f"ks1:={gains['ks1']}",
        "-p", f"ks2:={gains['ks2']}", "-p", f"phi1:={gains['phi1']}",
        "-p", f"phi2:={gains['phi2']}",
    ]
    if args.trajectory == "circle":
        controller_cmd.extend([
            "-p", f"radius:={args.radius}",
            "-p", f"yaw_bias_gain:={args.yaw_bias_gain}",
            "-p", f"radius_feedback_gain:={args.radius_feedback_gain}",
            "-p", f"radius_position_gain:={args.radius_position_gain}",
        ])
    elif args.trajectory == "eight":
        controller_cmd.extend(["-p", f"amplitude:={args.amplitude}", "-p", "periods:=1.0"])
    else:
        controller_cmd.extend([
            "-p", f"side_length:={args.side_length}",
            "-p", f"corner_radius:={args.corner_radius}",
            "-p", "max_w:=0.85",
        ])

    logger = controller = None
    status = "failed"
    failure = None
    try:
        logger = subprocess.Popen(logger_cmd, start_new_session=True)
        time.sleep(1.0)
        controller = subprocess.Popen(controller_cmd, start_new_session=True)
        print(
            f"RUNNING {args.trajectory} {controller_name} source={args.source} "
            f"for {duration:.1f} s; file={csv_path}"
        )
        print("Keep the terminal focused: Ctrl-C stops controller and logger safely.")
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if controller.poll() is not None:
                if controller.returncode == 0:
                    print("Controller completed trajectory normally.")
                    break
                raise RuntimeError(f"controller exited with error code {controller.returncode}")
            if logger.poll() is not None:
                raise RuntimeError(f"logger exited early with code {logger.returncode}")
            time.sleep(0.2)
        status = "complete"
    except KeyboardInterrupt:
        print("Interrupted by operator.")
        status = "interrupted"
    except Exception as exc:
        failure = exc
        print(f"Experiment failed: {exc}")
    finally:
        terminate_process(controller)
        terminate_process(logger)
        # Send zero velocity to stop the robot immediately
        try:
            subprocess.run(
                ["ros2", "topic", "pub", "--once", "/cmd_vel",
                 "geometry_msgs/msg/Twist",
                 '{"linear": {"x": 0.0}, "angular": {"z": 0.0}}'],
                timeout=3.0, capture_output=True,
            )
        except Exception:
            pass

    append_manifest(
        output_dir / "gain_manifest.csv",
        {
            "file": str(csv_path), "controller": controller_name,
            "trajectory": args.trajectory, "run_id": run_id, "source": args.source,
            "status": status,
            "odom_topic": odom_topic,
            "start_camera_x": start_pose[0], "start_camera_y": start_pose[1],
            "start_camera_yaw": start_pose[2],
            "k1": gains["k1"], "k2": gains["k2"], "k3": gains["k3"],
            "ks1": gains["ks1"], "ks2": gains["ks2"],
            "phi1": gains["phi1"], "phi2": gains["phi2"],
            "yaw_bias_gain": args.yaw_bias_gain,
            "radius_feedback_gain": args.radius_feedback_gain,
            "radius_position_gain": args.radius_position_gain,
        },
    )
    print(f"Saved run and gains to {output_dir / 'gain_manifest.csv'}")
    if failure is not None:
        raise failure
    return {
        "file": csv_path, "status": status, "gains": gains,
        "run_id": run_id, "start_pose": start_pose,
    }


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def rank_one(path, warmup, min_cmd_v, source="fusion"):
    rows = read_rows(path)
    active = []
    alignment_reference = None
    for row in rows:
        values = [number(row, key) for key in (
            "t", "error_ex", "error_ey", "error_etheta", "cmd_v", "cmd_w",
            "odom_x", "odom_y", "desired_x", "desired_y",
            "camera_x", "camera_y", "camera_yaw", "odom_yaw", "desired_yaw",
        )]
        if all(finite(value) for value in values):
            if alignment_reference is None:
                alignment_reference = values
            if values[0] >= warmup and values[4] > min_cmd_v:
                active.append(values)
    if len(active) < 20:
        raise ValueError("not enough active samples")
    ex = [row[1] for row in active]
    ey = [row[2] for row in active]
    etheta = [wrap_angle(row[3]) for row in active]
    cmd_v = [row[4] for row in active]
    cmd_w = [row[5] for row in active]
    ep = [math.hypot(row[8] - row[6], row[9] - row[7]) for row in active]
    # Put camera ground truth into the controller's initial world frame. This
    # is essential for raw-odometry runs because /odom_raw starts near zero
    # while the calibrated camera uses absolute floor coordinates.
    first = alignment_reference
    rotation = wrap_angle(first[13] - first[12])
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    camera_ep = []
    camera_heading_error = []
    for row in active:
        if source == "raw":
            dx_cam = row[10] - first[10]
            dy_cam = row[11] - first[11]
            camera_x = first[6] + cos_r * dx_cam - sin_r * dy_cam
            camera_y = first[7] + sin_r * dx_cam + cos_r * dy_cam
            camera_yaw = wrap_angle(first[13] + wrap_angle(row[12] - first[12]))
        else:
            camera_x = row[10]
            camera_y = row[11]
            camera_yaw = row[12]
        camera_ep.append(math.hypot(row[8] - camera_x, row[9] - camera_y))
        camera_heading_error.append(wrap_angle(row[14] - camera_yaw))
    dv = [cmd_v[index] - cmd_v[index - 1] for index in range(1, len(cmd_v))]
    dw = [cmd_w[index] - cmd_w[index - 1] for index in range(1, len(cmd_w))]
    rmse_ep = rmse(ep)
    rmse_heading = rmse(etheta)
    jerk_v = sample_std(dv)
    jerk_w = sample_std(dw)
    # Heuristic ranking only; raw metrics remain the scientific result.
    camera_rmse_ep = rmse(camera_ep)
    camera_rmse_heading = rmse(camera_heading_error)
    active_t = [row[0] - active[0][0] for row in active]
    convergence = math.nan
    hold_time = 2.0
    threshold = 0.05
    for index, start in enumerate(active_t):
        end = index
        while end < len(active_t) and active_t[end] <= start + hold_time:
            end += 1
        if end > index and active_t[end - 1] >= start + 0.9 * hold_time:
            if all(value <= threshold for value in camera_ep[index:end]):
                convergence = start
                break
    convergence_penalty = convergence if finite(convergence) else active_t[-1]
    score = (
        camera_rmse_ep
        + 0.20 * camera_rmse_heading
        + 0.002 * convergence_penalty
        + 0.10 * jerk_v
        + 0.03 * jerk_w
    )
    return {
        "file": str(path), "n_active": len(active),
        "active_duration_s": active[-1][0] - active[0][0],
        "rmse_ex_m": rmse(ex), "mean_ex_m": statistics.mean(ex),
        "rmse_ey_m": rmse(ey), "mean_ey_m": statistics.mean(ey),
        "rmse_etheta_rad": rmse_heading,
        "rmse_position_m": rmse_ep, "max_position_m": max(ep),
        "camera_aligned_rmse_position_m": camera_rmse_ep,
        "camera_aligned_max_position_m": max(camera_ep),
        "camera_aligned_rmse_heading_rad": camera_rmse_heading,
        "convergence_time_s": convergence,
        "cmd_v_delta_std": jerk_v, "cmd_w_delta_std": jerk_w,
        "score": score,
    }


OPTIMIZE_BOUNDS = {
    "k1": (0.20, 1.20), "k2": (1.0, 6.0), "k3": (1.5, 8.0),
    "ks1": (0.005, 0.30), "ks2": (0.005, 0.50),
    "phi1": (0.03, 1.20), "phi2": (0.05, 1.80),
}


def clamp(value, bounds):
    return max(bounds[0], min(bounds[1], value))


def propose_gains(rng, best, progress):
    if best is None or progress < 0.30:
        candidate = {}
        for key, bounds in OPTIMIZE_BOUNDS.items():
            if key.startswith("phi"):
                candidate[key] = math.exp(rng.uniform(math.log(bounds[0]), math.log(bounds[1])))
            else:
                candidate[key] = rng.uniform(*bounds)
        return candidate
    # Local stochastic search; radius shrinks as evidence accumulates.
    scale = max(0.08, 0.35 * (1.0 - progress))
    candidate = {}
    for key, bounds in OPTIMIZE_BOUNDS.items():
        span = bounds[1] - bounds[0]
        if key.startswith("phi"):
            value = best[key] * math.exp(rng.gauss(0.0, scale))
        else:
            value = best[key] + rng.gauss(0.0, scale * span)
        candidate[key] = clamp(value, bounds)
    return candidate


def aggregate_candidate(run_metrics):
    scores = [row["score"] for row in run_metrics]
    position = [row["camera_aligned_rmse_position_m"] for row in run_metrics]
    convergence = [
        row["convergence_time_s"] for row in run_metrics
        if finite(row["convergence_time_s"])
    ]
    return {
        "n_runs": len(run_metrics),
        "objective_mean": statistics.mean(scores),
        "objective_std": sample_std(scores),
        "objective_robust": statistics.mean(scores) + 0.5 * sample_std(scores),
        "camera_rmse_mean_m": statistics.mean(position),
        "camera_rmse_std_m": sample_std(position),
        "convergence_mean_s": statistics.mean(convergence) if convergence else math.nan,
        "worst_position_m": max(row["camera_aligned_max_position_m"] for row in run_metrics),
    }


def write_optimizer_history(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def optimize_circle(args):
    if args.controller != "bsmc":
        raise SystemExit("Automatic optimization currently supports --controller bsmc")
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "optimizer_history.csv"
    history = []
    best_gains = None
    best_objective = math.inf
    start_reference = None

    initial = dict(DEFAULT_GAINS["circle"])
    for candidate_index in range(args.iterations):
        gains = initial if candidate_index == 0 else propose_gains(
            rng, best_gains, candidate_index / max(1, args.iterations - 1)
        )
        metrics = []
        print(f"\n=== Candidate {candidate_index + 1}/{args.iterations}: {gains} ===")
        for repeat in range(args.repeats):
            run_args = argparse.Namespace(
                trajectory="circle", controller="bsmc", source=args.source,
                duration=args.duration, check_duration=args.check_duration,
                output_dir=str(output_dir),
                run_id=f"opt_c{candidate_index + 1:02d}_r{repeat + 1}",
                force=False, no_reset=False, radius=1.0, amplitude=0.5,
                side_length=1.0, corner_radius=0.12,
                yaw_bias_gain=0.0, radius_feedback_gain=0.0,
                radius_position_gain=0.0, start_reference=start_reference,
                start_position_tolerance=args.start_position_tolerance,
                start_yaw_tolerance_deg=args.start_yaw_tolerance_deg, **gains,
            )
            result = run_experiment(run_args)
            # Track drift between consecutive runs, not accumulated from first
            start_reference = result["start_pose"]
            if result["status"] != "complete":
                raise SystemExit("Optimization stopped because a run was not complete")
            metrics.append(rank_one(result["file"], args.warmup, 0.01, args.source))
            if args.cooldown > 0.0:
                print(f"Cooling down for {args.cooldown:.1f} s...")
                time.sleep(args.cooldown)
        aggregate = aggregate_candidate(metrics)
        row = {"candidate": candidate_index + 1, **gains, **aggregate}
        history.append(row)
        write_optimizer_history(history_path, history)
        if aggregate["objective_robust"] < best_objective:
            best_objective = aggregate["objective_robust"]
            best_gains = dict(gains)
            print(f"NEW BEST objective={best_objective:.6f}: {best_gains}")

    print("\n=== Validating best gain set ===")
    for repeat in range(args.validation_repeats):
        run_args = argparse.Namespace(
            trajectory="circle", controller="bsmc", source=args.source,
            duration=args.duration, check_duration=args.check_duration,
            output_dir=str(output_dir), run_id=f"best_validation_r{repeat + 1}",
            force=False, no_reset=False, radius=1.0, amplitude=0.5,
            side_length=1.0, corner_radius=0.12,
            yaw_bias_gain=0.0, radius_feedback_gain=0.0,
            radius_position_gain=0.0, start_reference=start_reference,
            start_position_tolerance=args.start_position_tolerance,
            start_yaw_tolerance_deg=args.start_yaw_tolerance_deg, **best_gains,
        )
        result = run_experiment(run_args)
        start_reference = result["start_pose"]
        if args.cooldown > 0.0 and repeat + 1 < args.validation_repeats:
            time.sleep(args.cooldown)
    print(f"Best gains: {best_gains}")
    print(f"Optimization history: {history_path}")


def rank_experiments(args):
    manifest = Path(args.manifest).expanduser().resolve()
    manifest_rows = read_rows(manifest)
    gains = {str(Path(row["file"]).resolve()): row for row in manifest_rows}
    results = []
    for file_name, gain_row in gains.items():
        if gain_row.get("status", "complete").strip().lower() != "complete":
            print(f"Skipping non-complete run: {file_name}")
            continue
        path = Path(file_name)
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        try:
            result = rank_one(
                path, args.warmup, args.min_cmd_v,
                source=gain_row.get("source", "fusion").strip().lower(),
            )
            for key in ("controller", "run_id", "k1", "k2", "k3", "ks1", "ks2", "phi1", "phi2"):
                result[key] = gain_row.get(key, "")
            results.append(result)
        except ValueError as exc:
            print(f"Skipping {path}: {exc}")
    results.sort(key=lambda row: row["score"])
    if not results:
        raise SystemExit("No valid runs to rank.")
    output = manifest.parent / "gain_ranking.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("rank,score,camera_rmse_cm,controller_rmse_cm,rmse_ex_cm,rmse_ey_cm,heading_deg,run_id")
    for index, row in enumerate(results, 1):
        print(
            f"{index},{row['score']:.5f},{100*row['camera_aligned_rmse_position_m']:.2f},"
            f"{100*row['rmse_position_m']:.2f},{100*row['rmse_ex_m']:.2f},{100*row['rmse_ey_m']:.2f},"
            f"{math.degrees(row['rmse_etheta_rad']):.3f},{row['run_id']}"
        )
    print(f"Full ranking: {output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="measure camera/EKF rate, delay, and static noise")
    check.add_argument("--duration", type=float, default=10.0)

    run = sub.add_parser("run", help="preflight, run one gain set, and log it")
    run.add_argument("--trajectory", choices=["circle", "eight", "square"], required=True)
    run.add_argument("--controller", choices=["bsmc", "backstepping"], required=True)
    run.add_argument("--source", choices=["fusion", "raw"], required=True)
    run.add_argument("--duration", type=float, default=0.0, help="0 uses one-trajectory default")
    run.add_argument("--check-duration", type=float, default=5.0)
    run.add_argument("--output-dir", default="paper_logs/gain_tuning")
    run.add_argument("--run-id", default="")
    run.add_argument("--force", action="store_true")
    run.add_argument("--no-reset", action="store_true", help="do not publish /reset_odom before starting")
    run.add_argument("--k1", type=float, default=None)
    run.add_argument("--k2", type=float, default=None)
    run.add_argument("--k3", type=float, default=None)
    run.add_argument("--ks1", type=float, default=None)
    run.add_argument("--ks2", type=float, default=None)
    run.add_argument("--phi1", type=float, default=None)
    run.add_argument("--phi2", type=float, default=None)
    run.add_argument("--radius", type=float, default=1.0)
    run.add_argument("--amplitude", type=float, default=0.5, help="figure-8 half-width; total width is 2A")
    run.add_argument("--side-length", type=float, default=1.0)
    run.add_argument(
        "--corner-radius", type=float, default=0.12,
        help="square corner radius; 0.12 m keeps desired yaw rate below bridge limit",
    )
    run.add_argument("--yaw-bias-gain", type=float, default=0.0)
    run.add_argument("--radius-feedback-gain", type=float, default=0.0)
    run.add_argument("--radius-position-gain", type=float, default=0.0)

    rank = sub.add_parser("rank", help="rank logged gain runs using active-motion samples")
    rank.add_argument("--manifest", default="paper_logs/gain_tuning/gain_manifest.csv")
    rank.add_argument("--warmup", type=float, default=5.0)
    rank.add_argument("--min-cmd-v", type=float, default=0.01)

    optimize = sub.add_parser("optimize", help="automatically search BSMC gains on a 1 m circle")
    optimize.add_argument("--controller", choices=["bsmc"], default="bsmc")
    optimize.add_argument("--source", choices=["fusion", "raw"], required=True)
    optimize.add_argument("--iterations", type=int, default=12)
    optimize.add_argument("--repeats", type=int, default=2)
    optimize.add_argument("--validation-repeats", type=int, default=5)
    optimize.add_argument("--duration", type=float, default=63.0)
    optimize.add_argument("--check-duration", type=float, default=5.0)
    optimize.add_argument("--warmup", type=float, default=5.0)
    optimize.add_argument("--cooldown", type=float, default=12.0)
    optimize.add_argument("--seed", type=int, default=7)
    optimize.add_argument("--output-dir", default="paper_logs/gain_optimization")
    optimize.add_argument("--start-position-tolerance", type=float, default=0.50)
    optimize.add_argument("--start-yaw-tolerance-deg", type=float, default=30.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "check":
        summary = collect_sensor_check(args.duration)
        print_sensor_summary(summary)
        problems = preflight_ok(summary)
        if problems:
            raise SystemExit("FAILED: " + "; ".join(problems))
        print("PASS: camera and filtered odometry are ready.")
    elif args.command == "run":
        run_experiment(args)
    elif args.command == "rank":
        rank_experiments(args)
    else:
        if args.iterations < 2 or args.repeats < 1 or args.validation_repeats < 1:
            raise SystemExit("iterations >= 2 and repeats/validation-repeats >= 1 are required")
        if args.start_position_tolerance <= 0.0 or args.start_yaw_tolerance_deg <= 0.0:
            raise SystemExit("start-pose tolerances must be positive")
        optimize_circle(args)


if __name__ == "__main__":
    main()
