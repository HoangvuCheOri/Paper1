"""Validated and experimental hardware profiles for circle tracking."""

from copy import deepcopy


CIRCLE_PROFILES = {
    # Baseline reproduces the current stable direct bsmc_circle controller.
    # Keep the sliding gains disabled until isolated hardware trials validate
    # them against this exact configuration.
    "1m": {
        "radius": 1.0,
        "angular_speed": 0.108,
        "k1": 1.163,
        "k2": 4.499,
        "k3": 3.300,
        "ks1": 0.0,
        "ks2": 0.0,
        "phi1": 1.0,
        "phi2": 1.5,
        "yaw_bias_gain": 0.02,
        "radius_feedback_gain": 0.60,
        "radius_position_gain": 0.40,
    },
}


def get_circle_profile(name):
    """Return an independent circle profile dictionary."""
    if name not in CIRCLE_PROFILES:
        choices = ", ".join(sorted(CIRCLE_PROFILES))
        raise ValueError(f"unknown circle profile {name!r}; choose {choices}")
    profile = deepcopy(CIRCLE_PROFILES[name])
    profile["name"] = name
    return profile
