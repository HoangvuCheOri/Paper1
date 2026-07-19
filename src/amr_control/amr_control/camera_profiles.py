#!/usr/bin/env python3
"""Explicit camera profiles for legacy and undistorted homographies."""

import os
from pathlib import Path

from rclpy.parameter import Parameter

from amr_control.camera import main as camera_main


def _find_workspace_file(filename):
    candidates = []
    configured_root = os.getenv("PAPER1_ROOT", "").strip()
    if configured_root:
        candidates.append(Path(configured_root).expanduser() / filename)
    candidates.append(Path.cwd() / filename)
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / filename)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find {filename}; searched: {searched}")


def _run_profile(filename, args=None):
    path = _find_workspace_file(filename)
    camera_main(
        args=args,
        parameter_overrides=[
            Parameter("floor_homography_path", value=path),
        ],
    )


def circle_square_main(args=None):
    """Run the camera with the unchanged distorted-domain homography."""
    _run_profile("floor_homography.yaml", args=args)


def eight_main(args=None):
    """Run the camera with the undistorted-domain figure-eight candidate."""
    _run_profile("floor_homography_undistorted_candidate.yaml", args=args)
