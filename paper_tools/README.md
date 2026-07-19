# BSMC paper asset pipeline

This directory generates publication figures and tables without importing or
modifying any robot controller. `datasets.yaml` is the frozen provenance
registry: every result in the main paper must point to one of its run IDs.

## Files

- `datasets.yaml`: selected runs, summaries, parameters, link tests, and image provenance.
- `paper_audit.py`: checks paths, schema, message freshness, square heading jumps,
  comparison coverage, and unavailable ESP-NOW timestamp/sequence fields.
- `paper_common.py`: CSV loading, freshness filtering, coordinate alignment, and
  camera-derived tracking errors.
- `paper_style.py`: consistent one/two-column plotting style and color mapping.
- `paper_figures.py`: Figures 2--6 as vector PDF and 600 dpi PNG.
- `paper_tables.py`: protocol, performance, controller-profile, and ESP-NOW
  tables in both CSV and LaTeX.
- `build_paper_assets.py`: audit-first entry point for the complete pipeline.

## Usage

From the repository root:

```bash
MPLCONFIGDIR=/tmp/bsmc-mpl python3 paper_tools/paper_audit.py
MPLCONFIGDIR=/tmp/bsmc-mpl python3 paper_tools/build_paper_assets.py \
  --output-dir paper_exports
```

To generate one figure only:

```bash
MPLCONFIGDIR=/tmp/bsmc-mpl python3 paper_tools/paper_figures.py \
  --only square --output-dir paper_exports/figures
```

## Scientific conventions

- Trajectories are transformed into a local frame using each run's first
  desired pose. This makes runs with different absolute camera origins
  comparable without altering their geometry.
- Figure 4 recomputes body-frame errors from raw AprilTag pose and desired pose;
  it does not treat the camera-fused EKF as independent ground truth.
- Square corner heading spikes are retained. They result from the exact
  polyline's discrete 90-degree desired-heading changes.
- ESP-NOW loss is labelled **gap-estimated**, because the registered logs have
  neither packet sequence numbers nor robot timestamps. One-way latency is not
  reported.
- Current performance rows are single-run results (`n=1`). Do not typeset
  mean/standard-deviation claims until independent repeats are registered.

Generated assets should be reviewed at final LaTeX size before submission.
The source experiment files remain untouched.

