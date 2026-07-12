# Paper Test Workflow

The BSMC paper needs repeatable data beyond a single robot run. Use these tests
to fill the TBD tables and generate paper figures.

## 1. Tracking comparison

Run each trajectory at least three times with the same initial setup:

- Backstepping-only: set `ks1:=0.0 ks2:=0.0`.
- BSMC: set `ks1` and `ks2` to the selected values.
- Trajectories: circle and figure-eight. Square can be reported as an extra
  validation if needed.

Start the logger during every run:

```bash
ros2 run amr_control paper_data_logger --ros-args \
  -p controller:=BSMC \
  -p trajectory:=circle \
  -p run_id:=bsmc_circle_01
```

Generate metrics and plots:

```bash
ros2 run amr_control paper_metrics tracking \
  --manifest paper_logs/manifest.csv \
  --source odom \
  --start-time 5.0 \
  --epsilon 0.05 \
  --hold-time 2.0 \
  --outdir paper_results/tracking
```

Use `--source camera` for AprilTag-ground-truth metrics when the camera data is
valid throughout the run.

## 2. Actuator bandwidth step test

Log a straight step and a spin step:

- Linear: `vcmd: 0 -> 0.15 m/s`, `wcmd = 0`.
- Angular: `wcmd: 0 -> 0.30 rad/s`, `vcmd = 0`.

Then compute the 63.2 percent time constant:

```bash
ros2 run amr_control paper_metrics step \
  --input paper_logs/linear_step.csv \
  --kind linear \
  --outdir paper_results/step

ros2 run amr_control paper_metrics step \
  --input paper_logs/angular_step.csv \
  --kind angular \
  --outdir paper_results/step
```

The paper can report `kv = 1/tau_v` and `kw = 1/tau_w`.

## 3. ESP-NOW link quality

Run the packet-level logger at each distance/condition:

```bash
ros2 run amr_control paper_link_logger --ros-args \
  -p condition:=static_5m \
  -p distance_m:=5.0
```

Analyze each CSV:

```bash
ros2 run amr_control paper_metrics link \
  --input paper_logs/espnow_static_5m.csv \
  --condition static_5m \
  --distance-m 5.0 \
  --outdir paper_results/link
```

If the firmware sends extended `DATA,robot_ms,seq,rpmL,rpmR,gyro` packets, the
script computes sequence-based loss. With the legacy `DATA,rpmL,rpmR,gyro`
format, it estimates loss from inter-arrival gaps only.

## 4. Odometry versus AprilTag validation

Use existing `odom_logger` or `paper_data_logger`, then compute drift metrics:

```bash
ros2 run amr_control paper_metrics localization \
  --input odom_compare.csv \
  --outdir paper_results/localization
```

Report RMSE position, final drift, max drift, and yaw RMSE.

## 5. Offline robustness ablation

Before robot experiments, generate reproducible simulation data:

```bash
ros2 run amr_control paper_bsmc_sim \
  --trajectory circle \
  --duration 80 \
  --linear-bias 0.005 \
  --angular-bias 0.01 \
  --disturbance-amp 0.01 \
  --outdir paper_results/sim_circle
```

This creates Backstepping and BSMC CSV files plus a manifest that can be passed
to `paper_metrics tracking`.
