import math

from dlp.waits import WaitProfile, blend_profile


def flat(value, n=96):
    return [float(value)] * n


def test_blend_matches_live_now_and_fades_to_prediction():
    pred = WaitProfile({1: flat(30)}, 15)
    now = 10 * 60 + 5
    b = blend_profile(pred, {1: {"wait": 70.0, "is_open": True}}, now, tau=60)
    assert b.wait(1, now) == 70                       # current slot = posted wait
    assert 30 < b.wait(1, now + 60) < 70              # an hour later: in between
    assert abs(b.wait(1, now + 6 * 60) - 30) < 1      # later in the day: prediction
    assert b.wait(1, 8 * 60) == 30                    # past slots untouched


def test_blend_closed_now_then_reopens():
    pred = WaitProfile({1: flat(20)}, 15)
    b = blend_profile(pred, {1: {"wait": 0.0, "is_open": False}}, 600, closed_minutes=60)
    assert math.isinf(b.wait(1, 600)) and math.isinf(b.wait(1, 650))
    assert b.wait(1, 700) == 20


def test_blend_open_when_model_expected_closed():
    pred = WaitProfile({1: [math.inf] * 96}, 15)
    b = blend_profile(pred, {1: {"wait": 15.0, "is_open": True}}, 600)
    assert b.wait(1, 600) == 15 and b.wait(1, 640) == 15
    assert math.isinf(b.wait(1, 700))
