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
from amr_control.bsmc_experiment import run_experiment, rank_one


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

LYAP = compute_lyapunov_bounds(DELTA_V_BAR, DELTA_OMEGA_BAR, margin=1.5)

# Search ranges cho k1, k2, k3 (không có closed-form Lyapunov bound)
K1_RANGE = (0.20, 1.20)
K2_RANGE = (1.0, 6.0)
K3_RANGE = (1.5, 8.0)

# phi1, phi2 defaults (boundary layer thickness — ít nhạy hơn)
PHI1_DEFAULT = 1.0
PHI2_DEFAULT = 1.5
PHI1_RANGE = (0.03, 1.20)
PHI2_RANGE = (0.05, 1.80)

# Penalty score cho trial thất bại
ABORT_PENALTY = 50.0


# ============================================================================
# 3. OPTUNA OBJECTIVE — GỌI ROBOT THẬT
# ============================================================================

def make_objective(args, output_dir, results_log):
    """Tạo objective function cho Optuna, mỗi call = 1 lần chạy robot thật."""

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

        # --- Chạy robot thật bằng run_experiment() ---
        run_args = argparse.Namespace(
            trajectory="circle",
            controller="bsmc",
            source=args.source,
            duration=args.duration,
            check_duration=5.0,
            output_dir=str(output_dir),
            run_id=f"optuna_t{trial.number + 1:03d}",
            force=False,
            no_reset=False,
            radius=1.0,
            amplitude=0.5,
            side_length=1.0,
            corner_radius=0.12,
            yaw_bias_gain=0.0,
            radius_feedback_gain=0.0,
            radius_position_gain=0.0,
            start_reference=None,        # Không cần — mỗi run neo vào vị trí hiện tại
            start_position_tolerance=0.50,
            start_yaw_tolerance_deg=30.0,
            **gains,
        )

        try:
            result = run_experiment(run_args)

            if result["status"] != "complete":
                print(f"  Trial {trial.number + 1}: INCOMPLETE → penalty {ABORT_PENALTY}")
                results_log.append({
                    "trial": trial.number + 1, "score": ABORT_PENALTY,
                    "status": "incomplete", **gains,
                })
                return ABORT_PENALTY

            # --- Tính metrics bằng rank_one() ---
            metrics = rank_one(
                result["file"], warmup=args.warmup,
                min_cmd_v=0.01, source=args.source,
            )
            score = metrics["score"]

            # Log chi tiết
            row = {
                "trial": trial.number + 1,
                **gains,
                "score": score,
                "rmse_position_m": metrics["rmse_position_m"],
                "max_position_m": metrics["max_position_m"],
                "rmse_etheta_deg": math.degrees(metrics["rmse_etheta_rad"]),
                "camera_rmse_position_m": metrics["camera_aligned_rmse_position_m"],
                "convergence_time_s": metrics.get("convergence_time_s", float("nan")),
                "cmd_v_delta_std": metrics.get("cmd_v_delta_std", 0),
                "cmd_w_delta_std": metrics.get("cmd_w_delta_std", 0),
                "n_active": metrics["n_active"],
                "file": str(result["file"]),
            }
            results_log.append(row)

            print(
                f"  Trial {trial.number + 1}: score={score:.6f}"
                f"  rmse_pos={metrics['rmse_position_m']:.4f}m"
                f"  rmse_θ={math.degrees(metrics['rmse_etheta_rad']):.2f}°"
                f"  max_pos={metrics['max_position_m']:.4f}m"
            )

        except (Exception, SystemExit) as e:
            print(f"  Trial {trial.number + 1}: ERROR «{e}» → penalty {ABORT_PENALTY}")
            results_log.append({
                "trial": trial.number + 1, "score": ABORT_PENALTY,
                "error": str(e), **gains,
            })
            score = ABORT_PENALTY

        # --- Cooldown giữa các trial ---
        if args.cooldown > 0:
            print(f"  Cooling down {args.cooldown:.0f}s...")
            time.sleep(args.cooldown)

        return score

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
                "trajectory": "circle",
                "source": args.source,
                "best_score": study.best_value,
                "best_params": study.best_params,
                "lyapunov_bounds": asdict(LYAP),
                "n_trials_completed": len(study.trials),
                "n_trials_requested": args.n_trials,
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


# ============================================================================
# 5. MAIN
# ============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Optuna Bayesian Optimization cho BSMC gains (hardware-in-the-loop)"
    )
    parser.add_argument("--source", choices=["fusion", "raw"], default="fusion")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Số trial (lần chạy robot thật). 20-40 thường đủ hội tụ.")
    parser.add_argument("--duration", type=float, default=63.0,
                        help="Thời gian mỗi trial (s)")
    parser.add_argument("--warmup", type=float, default=5.0,
                        help="Bỏ qua N giây đầu khi tính metrics")
    parser.add_argument("--cooldown", type=float, default=12.0,
                        help="Nghỉ giữa các trial (s) — cho robot dừng hẳn")
    parser.add_argument("--tune-phi", action="store_true",
                        help="Tối ưu luôn phi1, phi2 (mặc định: cố định)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="paper_logs/optuna_tuning")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Banner ---
    print("=" * 70)
    print("  BSMC Bayesian Optimization (Optuna TPE) — Hardware-in-the-Loop")
    print("=" * 70)
    print(f"  Source:     {args.source}")
    print(f"  Trials:     {args.n_trials}")
    print(f"  Duration:   {args.duration}s per trial")
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
        study_name="bsmc_circle_bayesian",
        direction="minimize",
        sampler=sampler,
    )

    objective = make_objective(args, output_dir, results_log)

    try:
        study.optimize(objective, n_trials=args.n_trials)
    except (KeyboardInterrupt, SystemExit):
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

    # Gợi ý bước tiếp theo
    print(f"\n  Bước tiếp: chạy validation với best gains:")
    best = study.best_params
    gain_args = " ".join(f"--{k} {v:.4f}" for k, v in best.items())
    print(f"  ros2 run amr_control bsmc_experiment run"
          f" --trajectory circle --controller bsmc --source {args.source} {gain_args}")


if __name__ == "__main__":
    main()
