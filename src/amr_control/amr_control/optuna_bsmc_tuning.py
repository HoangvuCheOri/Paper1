#!/usr/bin/env python3
"""
Hardware-in-the-loop Bayesian Optimization for BSMC gains using Optuna TPE.

Tích hợp hoàn toàn với bsmc_experiment: dùng run_experiment() để chạy robot thật
và rank_one() để tính cost function.

CÁCH DÙNG:
    ros2 run amr_control optuna_bsmc_tuning --source fusion --n-trials 30
    ros2 run amr_control optuna_bsmc_tuning --source fusion --n-trials 40 --tune-phi

PHƯƠNG PHÁP:
    - Optuna TPE (Tree-structured Parzen Estimator) thay thế random + local search.
    - Ks1, Ks2 search range dựa trên cận Lyapunov (margin=1.5 x disturbance bound).
    - Cost function J = rmse_position + 0.2*rmse_heading + 0.002*convergence + 0.1*jerk_v + 0.03*jerk_w
      (giống hệt score trong bsmc_experiment.rank_one).
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    raise SystemExit(
        "Optuna chưa được cài đặt. Chạy:\n"
        "  pip3 install optuna\n"
        "rồi thử lại."
    )

# Reuse existing infrastructure — chạy robot thật + tính metrics
from amr_control.bsmc_experiment import run_experiment, rank_one, DEFAULT_GAINS


# ============================================================================
# 1. CẬN LYAPUNOV CHO Ks1, Ks2
# ============================================================================

@dataclass
class LyapunovBounds:
    """Search bounds for Ks derived from Lyapunov stability condition."""
    ks1_min: float
    ks2_min: float
    ks1_max: float
    ks2_max: float


def compute_lyapunov_bounds(
    delta_v_bar: float,
    delta_omega_bar: float,
    margin: float = 1.5,
    ceiling_factor: float = 4.0,
) -> LyapunovBounds:
    """
    Tính khoảng tìm kiếm Ks từ cận nhiễu đo được.

    Theo Theorem (BSMC stability): Ks_i > |delta_i_bar| để V_dot < 0 ngoài
    boundary layer.  margin nhân thêm hệ số an toàn.

    Parameters
    ----------
    delta_v_bar : float
        Cận nhiễu |Δv| đo từ step-response residual [m/s].
    delta_omega_bar : float
        Cận nhiễu |Δω| đo từ step-response residual [rad/s].
    margin : float
        Hệ số an toàn (1.3–2.0 hợp lý cho hardware).
    ceiling_factor : float
        Trần trên = ks_min * ceiling_factor, tránh chattering vô ích.
    """
    ks1_min = margin * delta_v_bar
    ks2_min = margin * delta_omega_bar
    return LyapunovBounds(
        ks1_min=ks1_min,
        ks2_min=ks2_min,
        ks1_max=ks1_min * ceiling_factor,
        ks2_max=ks2_min * ceiling_factor,
    )


# ============================================================================
# 2. CẤU HÌNH MẶC ĐỊNH
# ============================================================================

# Cận nhiễu đo từ step-response test (Section V.A)
# TODO: Cập nhật từ dữ liệu step-response thực tế của bạn
DELTA_V_BAR = 0.015        # [m/s]
DELTA_OMEGA_BAR = 0.05     # [rad/s]

LYAP = compute_lyapunov_bounds(DELTA_V_BAR, DELTA_OMEGA_BAR, margin=1.5, ceiling_factor=12.0)

# Search ranges cho k1, k2, k3 (đã nới rộng, Optuna TPE sẽ tự động thu hẹp vùng tìm kiếm)
K1_RANGE = (0.30, 1.50)
K2_RANGE = (5.00, 12.00)
K3_RANGE = (3.00, 9.00)

# phi1, phi2 defaults (boundary layer thickness — ít nhạy hơn)
PHI1_DEFAULT = 1.0
PHI2_DEFAULT = 1.5
PHI1_RANGE = (0.03, 1.20)
PHI2_RANGE = (0.05, 1.80)

# Penalty score cho trial thất bại
ABORT_PENALTY = 50.0


def metric_score(metrics, heading_weight=0.10):
    """Accuracy-first score; all terms are dimensionless after weighting."""
    return (
        metrics["camera_aligned_rmse_position_m"]
        + 0.25 * metrics["camera_aligned_max_position_m"]
        + heading_weight * metrics["camera_aligned_rmse_heading_rad"]
        + 0.001 * metrics["convergence_penalty_s"]
        + 0.05 * metrics["cmd_v_delta_std"]
        + 0.02 * metrics["cmd_w_delta_std"]
        + 0.10 * metrics["command_saturation_fraction"]
    )


def build_run_args(args, gains, output_dir, run_id):
    return argparse.Namespace(
        trajectory=args.trajectory, controller="bsmc", source=args.source,
        duration=args.duration, check_duration=5.0,
        output_dir=str(output_dir), run_id=run_id, force=False, no_reset=False,
        radius=1.0, amplitude=args.amplitude,
        angular_speed=args.angular_speed, side_length=1.0,
        corner_radius=0.1, yaw_bias_gain=0.0,
        radius_feedback_gain=0.0, radius_position_gain=0.0,
        start_reference=None, start_position_tolerance=0.50,
        start_yaw_tolerance_deg=30.0, **gains,
    )


def wait_for_reposition(trial_number, args):
    """Require a human safety check before a hardware trial."""
    if not args.require_confirmation:
        return
    if not sys.stdin.isatty():
        raise SystemExit(
            "Reposition confirmation requires an interactive terminal. "
            "Use --auto-continue only when the test area is safe."
        )
    print("\nSAFETY GATE")
    print(f"  Trial {trial_number}: put the robot at the marked start pose.")
    print("  Check that the whole figure-8 is clear and the AprilTag is visible.")
    answer = input("  Press Enter to run, or type q to stop: ").strip().lower()
    if answer in {"q", "quit", "stop"}:
        raise KeyboardInterrupt


# ============================================================================
# 3. OPTUNA OBJECTIVE — GỌI ROBOT THẬT
# ============================================================================

def make_objective(args, output_dir, results_log):
    """Each Optuna candidate is evaluated over repeated hardware runs."""

    def objective(trial: optuna.Trial) -> float:
        # --- Đề xuất gain từ Optuna TPE ---
        gains = {
            "k1": trial.suggest_float("k1", *K1_RANGE),
            "k2": trial.suggest_float("k2", *K2_RANGE),
            "k3": trial.suggest_float("k3", *K3_RANGE),
            # Ks1, Ks2: search trong [Lyapunov lower bound, ceiling]
            "ks1": trial.suggest_float("ks1", LYAP.ks1_min, LYAP.ks1_max),
            "ks2": trial.suggest_float("ks2", LYAP.ks2_min, LYAP.ks2_max),
        }
        if args.tune_phi:
            gains["phi1"] = trial.suggest_float("phi1", *PHI1_RANGE)
            gains["phi2"] = trial.suggest_float("phi2", *PHI2_RANGE)
        else:
            gains["phi1"] = PHI1_DEFAULT
            gains["phi2"] = PHI2_DEFAULT

        print(f"\n{'='*70}")
        print(f"Trial {trial.number + 1}/{args.n_trials}: {gains}")
        print(f"{'='*70}")

        run_scores = []
        run_metrics = []
        for repeat in range(1, args.repeats + 1):
            wait_for_reposition(f"{trial.number + 1}.{repeat}", args)
            run_id = f"optuna_t{trial.number + 1:03d}_r{repeat:02d}"
            try:
                result = run_experiment(
                    build_run_args(args, gains, output_dir, run_id)
                )
                if result["status"] != "complete":
                    raise RuntimeError(f"run status={result['status']}")
                metrics = rank_one(
                    result["file"], warmup=args.warmup,
                    min_cmd_v=0.01, source=args.source,
                )
                score = metric_score(metrics, args.heading_weight)
                run_scores.append(score)
                run_metrics.append(metrics)
                results_log.append({
                    "result_type": "run", "trial": trial.number + 1,
                    "repeat": repeat, **gains, "score": score,
                    "camera_rmse_position_m": metrics["camera_aligned_rmse_position_m"],
                    "camera_max_position_m": metrics["camera_aligned_max_position_m"],
                    "camera_rmse_heading_deg": math.degrees(
                        metrics["camera_aligned_rmse_heading_rad"]
                    ),
                    "convergence_time_s": metrics["convergence_time_s"],
                    "command_saturation_fraction": metrics["command_saturation_fraction"],
                    "trajectory_closure_error_m": metrics["trajectory_closure_error_m"],
                    "file": str(result["file"]),
                })
                print(
                    f"  Run {repeat}/{args.repeats}: score={score:.5f}, "
                    f"camera RMSE={100*metrics['camera_aligned_rmse_position_m']:.2f}cm, "
                    f"max={100*metrics['camera_aligned_max_position_m']:.2f}cm"
                )
            except (Exception, SystemExit) as exc:
                print(f"  Run {repeat}/{args.repeats}: ERROR «{exc}»")
                results_log.append({
                    "result_type": "run", "trial": trial.number + 1,
                    "repeat": repeat, **gains, "score": ABORT_PENALTY,
                    "error": str(exc),
                })
                run_scores.append(ABORT_PENALTY)
            if args.cooldown > 0:
                time.sleep(args.cooldown)

        mean_score = statistics.mean(run_scores)
        score_std = statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0
        worst_error = max(
            (m["camera_aligned_max_position_m"] for m in run_metrics),
            default=float("nan"),
        )
        worst_closure = max(
            (m["trajectory_closure_error_m"] for m in run_metrics),
            default=float("nan"),
        )
        worst_heading_deg = max(
            (math.degrees(m["camera_aligned_rmse_heading_rad"]) for m in run_metrics),
            default=float("nan"),
        )
        robust_score = mean_score + args.std_weight * score_std
        rejected = (
            not run_metrics
            or worst_error > args.max_error
            or worst_closure > args.max_closure_error
            or worst_heading_deg > args.max_heading_rmse_deg
        )
        if rejected:
            rejection_penalty = (
                5.0 + worst_error if math.isfinite(worst_error)
                else ABORT_PENALTY
            )
            robust_score = max(robust_score, rejection_penalty)
        results_log.append({
            "result_type": "candidate", "trial": trial.number + 1,
            "repeat": "all", **gains, "score": robust_score,
            "mean_score": mean_score, "score_std": score_std,
            "worst_camera_error_m": worst_error,
            "worst_trajectory_closure_m": worst_closure,
            "worst_camera_heading_rmse_deg": worst_heading_deg,
            "rejected_max_error": rejected,
        })
        print(
            f"  Candidate robust score={robust_score:.5f} "
            f"(mean={mean_score:.5f}, std={score_std:.5f}, "
            f"worst={100*worst_error:.1f}cm, "
            f"closure={100*worst_closure:.1f}cm, rejected={rejected})"
        )
        return robust_score

    return objective


# ============================================================================
# 4. SAVE RESULTS
# ============================================================================

def save_results(study, results_log, output_dir, args):
    """Lưu kết quả cho paper: best_params.json + all_trials.csv + param_importance."""

    # Best params
    best_path = output_dir / "best_params.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "method": "Bayesian Optimization (Optuna TPE)",
                "trajectory": args.trajectory,
                "source": args.source,
                "best_score": study.best_value,
                "best_params": study.best_params,
                "lyapunov_bounds": asdict(LYAP),
                "n_trials_completed": len(study.trials),
                "n_trials_requested": args.n_trials,
                "repeats_per_candidate": args.repeats,
                "validation_repeats": args.validation_repeats,
                "robust_score_std_weight": args.std_weight,
                "maximum_allowed_error_m": args.max_error,
                "maximum_trajectory_closure_error_m": args.max_closure_error,
                "heading_weight": args.heading_weight,
                "maximum_heading_rmse_deg": args.max_heading_rmse_deg,
                "tune_phi": args.tune_phi,
                "duration_per_trial_s": args.duration,
            },
            f,
            indent=2,
        )
    print(f"  Best params saved: {best_path}")

    # All trials CSV
    trials_csv = output_dir / "all_trials.csv"
    if results_log:
        all_keys = sorted({k for row in results_log for k in row})
        with open(trials_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for row in results_log:
                writer.writerow(row)
        print(f"  All trials saved: {trials_csv}")

    # Param importance (rất hữu ích cho Discussion section)
    try:
        importances = optuna.importance.get_param_importances(study)
        imp_path = output_dir / "param_importance.json"
        with open(imp_path, "w") as f:
            json.dump(importances, f, indent=2)
        print("\n  Param importance (ảnh hưởng lên cost J):")
        for param, imp in importances.items():
            bar = "█" * int(imp * 40)
            print(f"    {param:>4s}: {imp:.3f}  {bar}")
    except Exception:
        pass


def validate_best(args, best_params, output_dir):
    """Run independent repetitions that do not influence Optuna selection."""
    if args.validation_repeats <= 0:
        return
    gains = dict(best_params)
    gains.setdefault("phi1", PHI1_DEFAULT)
    gains.setdefault("phi2", PHI2_DEFAULT)
    rows = []
    print(f"\nIndependent validation: {args.validation_repeats} runs")
    for repeat in range(1, args.validation_repeats + 1):
        wait_for_reposition(f"validation.{repeat}", args)
        run_id = f"validation_best_r{repeat:02d}"
        try:
            result = run_experiment(
                build_run_args(args, gains, output_dir, run_id)
            )
            metrics = rank_one(
                result["file"], warmup=args.warmup,
                min_cmd_v=0.01, source=args.source,
            )
            rows.append({
                "repeat": repeat, "status": result["status"],
                "score": metric_score(metrics, args.heading_weight),
                "camera_rmse_position_m": metrics["camera_aligned_rmse_position_m"],
                "camera_max_position_m": metrics["camera_aligned_max_position_m"],
                "camera_rmse_heading_deg": math.degrees(
                    metrics["camera_aligned_rmse_heading_rad"]
                ),
                "convergence_time_s": metrics["convergence_time_s"],
                "file": str(result["file"]),
            })
        except (Exception, SystemExit) as exc:
            rows.append({"repeat": repeat, "status": "failed", "error": str(exc)})
        if args.cooldown > 0 and repeat < args.validation_repeats:
            time.sleep(args.cooldown)
    fields = sorted({key for row in rows for key in row})
    path = output_dir / "validation_runs.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if "camera_rmse_position_m" in row]
    if valid:
        rmse_values = [row["camera_rmse_position_m"] for row in valid]
        max_values = [row["camera_max_position_m"] for row in valid]
        print(
            f"Validation camera RMSE={100*statistics.mean(rmse_values):.2f}cm "
            f"± {100*(statistics.stdev(rmse_values) if len(rmse_values)>1 else 0):.2f}cm; "
            f"worst error={100*max(max_values):.2f}cm"
        )
    print(f"Validation saved: {path}")


# ============================================================================
# 5. MAIN
# ============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Optuna Bayesian Optimization cho BSMC gains (hardware-in-the-loop)"
    )
    parser.add_argument("--trajectory", choices=["circle", "eight", "square"],
                        default="circle", help="Quỹ đạo cần tune")
    parser.add_argument("--source", choices=["fusion", "raw"], default="fusion")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Số candidate gain Optuna (mỗi candidate chạy --repeats lần)")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Số lần chạy robot cho mỗi candidate")
    parser.add_argument("--validation-repeats", type=int, default=3,
                        help="Số lần validation độc lập cho best gain")
    parser.add_argument("--std-weight", type=float, default=0.5,
                        help="Trọng số phạt độ lệch chuẩn giữa các repeat")
    parser.add_argument("--max-error", type=float, default=0.12,
                        help="Loại candidate nếu peak camera error vượt ngưỡng này (m)")
    parser.add_argument("--max-closure-error", type=float, default=0.03,
                        help="Loại run chưa hoàn thành quỹ đạo kín (m)")
    parser.add_argument("--heading-weight", type=float, default=0.10,
                        help="Trọng số RMSE góc trong objective")
    parser.add_argument("--max-heading-rmse-deg", type=float, default=8.0,
                        help="Loại candidate nếu heading RMSE vượt ngưỡng")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Thời gian mỗi trial (s). 0 = tự chọn theo trajectory.")
    parser.add_argument("--warmup", type=float, default=5.0,
                        help="Bỏ qua N giây đầu khi tính metrics")
    parser.add_argument("--cooldown", type=float, default=12.0,
                        help="Nghỉ giữa các trial (s) — cho robot dừng hẳn")
    parser.add_argument("--amplitude", type=float, default=0.50,
                        help="Nửa chiều rộng figure-8 (m); tổng rộng xấp xỉ 2A")
    parser.add_argument("--angular-speed", type=float, default=0.10,
                        help="Tốc độ pha yêu cầu cho figure-8 (rad/s)")
    parser.add_argument("--require-confirmation", action="store_true",
                        help="Chờ người dùng xác nhận trước mỗi trial (mặc định chạy liên tục)")
    parser.add_argument("--tune-phi", action="store_true",
                        help="Tối ưu luôn phi1, phi2 (mặc định: cố định)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="paper_logs/optuna_tuning")
    args = parser.parse_args(argv)

    if args.n_trials <= 0:
        parser.error("--n-trials must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.validation_repeats < 0:
        parser.error("--validation-repeats must be >= 0")
    if args.std_weight < 0.0:
        parser.error("--std-weight must be >= 0")
    if args.max_error <= 0.0:
        parser.error("--max-error must be positive")
    if args.max_closure_error <= 0.0:
        parser.error("--max-closure-error must be positive")
    if args.heading_weight < 0.0:
        parser.error("--heading-weight must be >= 0")
    if args.max_heading_rmse_deg <= 0.0:
        parser.error("--max-heading-rmse-deg must be positive")
    if args.angular_speed <= 0.0:
        parser.error("--angular-speed must be positive")
    if args.amplitude <= 0.0:
        parser.error("--amplitude must be positive")

    # Auto-select duration if not specified
    # Tính chính xác cho square dựa trên thông số quỹ đạo:
    #   lead_in = corner_radius
    #   loop_length = 4*(side_length - 2*corner_radius) + 4*(pi/2*corner_radius)
    #   total_distance = lead_in + n_laps * loop_length
    #   t_track = total_distance / VD + T_ramp/2    (T_ramp = 2.5s)
    #   duration = t_track + startup_delay + settle_time + ROS_init_overhead
    SQUARE_N_LAPS = 2
    SQUARE_VD = 0.10
    SQUARE_SIDE = 1.0
    SQUARE_CR = 0.1
    SQUARE_RAMP = 2.5
    SQUARE_OVERHEAD = 2.0   # 1s(startup) + 2s(settle) - 1s(overlap)
    sq_loop = 4.0 * (SQUARE_SIDE - 2*SQUARE_CR) + 4.0 * (math.pi/2 * SQUARE_CR)
    sq_dist = SQUARE_CR + SQUARE_N_LAPS * sq_loop
    sq_t_track = sq_dist / SQUARE_VD + SQUARE_RAMP / 2.0
    sq_duration = sq_t_track + SQUARE_OVERHEAD
    # Controller limits figure-8 reference speed to 60% of max_v. Include
    # startup + settling + one complete phase cycle + ramp tail + margin.
    effective_w = min(
        args.angular_speed,
        0.18 * 0.60 / (args.amplitude * math.sqrt(2.0)),
    )
    eight_duration = 1.0 + 2.0 + 2.0 * math.pi / effective_w + 1.25 + 2.0
    DURATIONS = {"circle": 63.0, "eight": eight_duration, "square": sq_duration}
    if args.duration <= 0:
        args.duration = DURATIONS.get(args.trajectory, 63.0)

    output_dir = Path(args.output_dir).expanduser().resolve() / args.trajectory
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Banner ---
    print("=" * 70)
    print(f"  BSMC Optuna TPE — trajectory: {args.trajectory.upper()}")
    print("=" * 70)
    print(f"  Source:     {args.source}")
    print(f"  Candidates: {args.n_trials} x {args.repeats} repeats")
    print(f"  Validation: {args.validation_repeats} independent runs")
    print(f"  Duration:   {args.duration}s per trial")
    print(f"  Amplitude:  {args.amplitude}m (figure-8 width ≈ {2*args.amplitude:.2f}m)")
    print(f"  Tune phi:   {args.tune_phi}")
    print(f"  Cooldown:   {args.cooldown}s")
    print(f"  Output:     {output_dir}")
    print(f"  Ks1 range:  [{LYAP.ks1_min:.4f}, {LYAP.ks1_max:.4f}]  (Lyapunov, margin=1.5)")
    print(f"  Ks2 range:  [{LYAP.ks2_min:.4f}, {LYAP.ks2_max:.4f}]  (Lyapunov, margin=1.5)")
    print("=" * 70)
    print("  Nhấn Ctrl-C bất kỳ lúc nào để dừng — kết quả đã thu được sẽ được lưu.")
    print("  Bạn có thể nhấc robot về giữa phòng giữa các trial (chỉ WARNING, không crash).")
    print("=" * 70)

    # --- Optuna study ---
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results_log = []
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=f"bsmc_{args.trajectory}_bayesian",
        direction="minimize",
        sampler=sampler,
    )

    # First reference trial. Ks values are clipped to the declared Lyapunov
    # search domain so Optuna never evaluates a point outside its own bounds.
    baseline = DEFAULT_GAINS[args.trajectory]
    enqueued_params = {
        "k1": baseline["k1"],
        "k2": baseline["k2"],
        "k3": baseline["k3"],
        "ks1": baseline["ks1"],
        "ks2": baseline["ks2"],
    }
    if args.tune_phi:
        enqueued_params["phi1"] = baseline["phi1"]
        enqueued_params["phi2"] = baseline["phi2"]

    # Optuna có thể báo lỗi nếu giá trị default nằm ngoài bounds, ta kẹp nó lại
    enqueued_params["ks1"] = max(LYAP.ks1_min, min(LYAP.ks1_max, enqueued_params["ks1"]))
    enqueued_params["ks2"] = max(LYAP.ks2_min, min(LYAP.ks2_max, enqueued_params["ks2"]))

    study.enqueue_trial(enqueued_params)

    objective = make_objective(args, output_dir, results_log)

    interrupted = False
    try:
        study.optimize(objective, n_trials=args.n_trials)
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        print(f"\n\nDừng sớm sau {len(study.trials)} trials.")

    # --- Kết quả ---
    if not study.trials:
        print("Không có trial nào hoàn thành.")
        return

    print("\n" + "=" * 70)
    print(f"  HOÀN TẤT — {len(study.trials)} trials")
    print(f"  Best score:  {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")
    print("=" * 70)

    save_results(study, results_log, output_dir, args)
    if interrupted:
        print("Batch was interrupted; skipping automatic validation.")
        return
    if study.best_value >= 5.0:
        print("All completed candidates were rejected; skipping validation.")
        return
    validate_best(args, study.best_params, output_dir)

    # Gợi ý bước tiếp theo
    print(f"\n  Bước tiếp: chạy validation với best gains:")
    best = study.best_params
    gain_args = " ".join(f"--{k} {v:.4f}" for k, v in best.items())
    print(f"  ros2 run amr_control bsmc_experiment run"
          f" --trajectory {args.trajectory} --controller bsmc --source {args.source} {gain_args}")


if __name__ == "__main__":
    main()
