"""Validated hardware profiles for continuous sharp-corner squares."""

from copy import deepcopy


COMMON_SQUARE_PROFILE = {
    "corner_speed": 0.04,
    "desired_speed": 0.10,
    "min_v": 0.02,
    "max_w": 0.85,
    "sharp_corner_w": 0.55,
    "sharp_corner_blend_full_deg": 35.0,
    "k3": 7.0,
}


SQUARE_PROFILES = {
    # Best balanced 1 m run: path RMS 1.58 cm, ripple 0.67 cm,
    # straight heading RMS 1.91 deg.
    "1m": {
        "side_length": 1.0,
        "corner_decel_distance": 0.13,
        "sharp_corner_blend_start_deg": 12.0,
        "k2": 4.0,
        "kd_w": 0.0,
        "kd_w_deadband": 0.20,
    },
    # Accepted 2 m run: path RMS 2.88 cm, ripple 0.90 cm,
    # straight heading RMS 1.93 deg.
    "2m": {
        "side_length": 2.0,
        "corner_decel_distance": 0.25,
        "sharp_corner_blend_start_deg": 8.0,
        "k2": 6.0,
        "kd_w": 0.18,
        "kd_w_deadband": 0.20,
    },
}


def get_square_profile(name):
    """Return an independent, complete profile dictionary."""
    if name not in SQUARE_PROFILES:
        choices = ", ".join(sorted(SQUARE_PROFILES))
        raise ValueError(f"unknown square profile {name!r}; choose {choices}")
    profile = deepcopy(COMMON_SQUARE_PROFILE)
    profile.update(SQUARE_PROFILES[name])
    profile["name"] = name
    return profile


def profile_name_for_side(side_length):
    """Map the two validated side lengths to their named profile."""
    side = float(side_length)
    for name, values in SQUARE_PROFILES.items():
        if abs(side - values["side_length"]) <= 1e-6:
            return name
    raise ValueError(
        f"no validated profile for side_length={side:g}; use 1.0 or 2.0"
    )
