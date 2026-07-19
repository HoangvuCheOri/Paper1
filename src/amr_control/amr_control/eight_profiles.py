"""Validated hardware profiles for horizontal figure-8 tracking."""

from copy import deepcopy


EIGHT_PROFILES = {
    # Horizontal 1 m hardware preset. The 2026-07-18 baseline gave overall
    # position/path RMS 3.39/1.84 cm. Directional crossing analysis found
    # 0.69 cm lateral RMS left-to-right but 2.94 cm right-to-left, so a
    # centre-local k2 adds cross-track correction only near the crossing.
    # The first k2=10/radius=0.50 trial left a 2.80 cm right-to-left lateral
    # RMS because the offset was already present at the edge of that region;
    # start the smooth correction earlier and use the available yaw headroom.
    "1m": {
        "amplitude": 1.0,
        "angular_speed": 0.07,
        "trajectory_ramp_time": 12.0,
        # The standard path starts at its centre with a +45 deg tangent.
        # Rotating it -45 deg preserves the centre start and aligns that
        # tangent with a robot pose of (0, 0, 0).
        "path_rotation_deg": -45.0,
        "start_phase_deg": 0.0,
        "entry_heading_blend_time": 0.0,
        "initial_align_time": 0.0,
        "initial_align_timeout": 12.0,
        "w_feedforward_scale": 0.80,
        # A 0.70 trial reduced heading RMS but doubled negative-lobe lateral
        # bias (1.43 -> 2.99 cm) and raised path RMS (1.79 -> 2.44 cm).
        # Preserve the position/path-optimal validated value.
        "w_feedforward_scale_negative": 0.55,
        "w_feedforward_scale_positive": 0.80,
        "negative_yaw_rate_feedback_gain": 0.50,
        "positive_yaw_rate_feedback_gain": 0.30,
        "feedback_speed_floor": 0.05,
        "center_k1": 1.5,
        "center_k1_radius": 0.50,
        "center_k2": 18.0,
        "center_k2_radius": 0.65,
        # At the right-to-left crossing, k2*e_y (+0.0315 rad/s) and
        # k3*sin(e_theta) (-0.0364 rad/s) cancelled. Lower k3 only near the
        # centre so the robot may steer across the residual parallel offset.
        # With k3=3.5 the right-to-left centre terms still cancelled. Reducing
        # it to 2.0 improved that branch, but run 20260719_170630 still showed
        # ey=+0.96 cm against e_theta=-3.51 deg at the crossing, leaving only
        # about 0.005 rad/s net steering. Use 1.0 to favour lateral recovery
        # near the centre while leaving the outer-lobe k3=7 unchanged.
        "center_k3": 1.0,
        "center_k3_radius": 0.65,
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
