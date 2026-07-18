from amr_control.circle_profiles import get_circle_profile


def test_one_metre_circle_profile_is_complete():
    profile = get_circle_profile("1m")
    assert profile["radius"] == 1.0
    assert profile["angular_speed"] == 0.108
    assert profile["k1"] == 1.163
    assert profile["k2"] == 4.499
    assert profile["k3"] == 3.300
    assert profile["ks1"] == 0.0
    assert profile["ks2"] == 0.0
    assert profile["phi1"] > 0.0
    assert profile["phi2"] > 0.0


def test_circle_profile_returns_a_copy():
    first = get_circle_profile("1m")
    first["ks2"] = 99.0
    assert get_circle_profile("1m")["ks2"] == 0.0
