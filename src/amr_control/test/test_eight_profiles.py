from amr_control.eight_profiles import get_eight_profile


def test_one_metre_profile_is_complete():
    profile = get_eight_profile("1m")
    assert profile["amplitude"] == 1.0
    assert profile["angular_speed"] == 0.07
    assert profile["trajectory_ramp_time"] == 12.0
    assert profile["k2"] == 6.5
    assert profile["k3"] == 7.0
    assert profile["entry_heading_blend_time"] == 2.0
    assert profile["w_feedforward_scale_negative"] == 0.55
    assert profile["w_feedforward_scale_positive"] == 0.80
    assert profile["negative_yaw_rate_feedback_gain"] == 0.50
    assert profile["positive_yaw_rate_feedback_gain"] == 0.30
    assert profile["feedback_speed_floor"] == 0.05
    assert profile["center_k1"] == -1.0
    assert profile["center_k1_radius"] == 0.30
    assert profile["initial_align_time"] == 0.0
    assert profile["initial_align_timeout"] >= profile["initial_align_time"]


def test_profile_result_is_independent():
    first = get_eight_profile("1m")
    first["k2"] = -1.0
    assert get_eight_profile("1m")["k2"] == 6.5
