"""Validated hardware profiles for horizontal figure-8 tracking."""

from copy import deepcopy


EIGHT_PROFILES = {
    # Final horizontal 1 m hardware preset. Centre-local k1=1.5 with radius
    # 0.50 m gave position/path RMS 3.39/1.84 cm, waviness 0.09 cm, heading
    # RMS 5.9 deg, symmetry 2.46 cm, near-centre symmetry 2.71 cm, and
    # crossing position/path 2.66/2.01 cm.
    "1m": {
        "amplitude": 1.0,
        "angular_speed": 0.07,
        "trajectory_ramp_time": 12.0,
        "path_rotation_deg": 0.0,
        "entry_heading_blend_time": 2.0,
        # Operator preference: start moving immediately from yaw=0. The
        # optional alignment mode remains available as a CLI override.
        "initial_align_time": 0.0,
        "initial_align_timeout": 12.0,
        "w_feedforward_scale": 0.80,
        "w_feedforward_scale_negative": 0.55,
        "w_feedforward_scale_positive": 0.80,
        "negative_yaw_rate_feedback_gain": 0.50,
        "positive_yaw_rate_feedback_gain": 0.30,
        "feedback_speed_floor": 0.05,
        "center_k1": 1.5,
        "center_k1_radius": 0.50,
        "k1": 0.2205844943,
        "k2": 6.5,
        "k3": 7.0,
        "ks1": 0.03683289,
        "ks2": 0.1159106176,
        "phi1": 1.0,
        "phi2": 1.5,
    },
}


def get_eight_profile(name):
    """Return an independent figure-8 profile dictionary."""
    if name not in EIGHT_PROFILES:
        choices = ", ".join(sorted(EIGHT_PROFILES))
        raise ValueError(f"unknown figure-8 profile {name!r}; choose {choices}")
    profile = deepcopy(EIGHT_PROFILES[name])
    profile["name"] = name
    return profile
