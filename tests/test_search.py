import math
import random

import pytest

from dlp.park import START, load_rides, walk_minutes
from dlp.search import Infeasible, plan_astar, plan_brute_force, plan_greedy, plan_ucs
from dlp.waits import WaitProfile, constant_profile

RIDE_IDS = sorted(r.id for r in load_rides().values() if not r.single_rider)


def random_profile(rng, ride_ids, slot=15):
    n = 24 * 60 // slot
    slots = {}
    for rid in ride_ids:
        base = rng.choice([5, 15, 30, 60])
        slots[rid] = [max(0, base + rng.randint(-base, base)) for _ in range(n)]
    return WaitProfile(slots, slot)


@pytest.mark.parametrize("seed", range(25))
def test_astar_matches_brute_force(seed):
    rng = random.Random(seed)
    picks = rng.sample(RIDE_IDS, rng.randint(2, 6))
    profile = random_profile(rng, picks)
    start = 8 * 60 + rng.randint(0, 120)
    best = plan_brute_force(picks, start, profile)
    a = plan_astar(picks, start, profile)
    u = plan_ucs(picks, start, profile)
    assert a.finish == pytest.approx(best.finish)
    assert u.finish == pytest.approx(best.finish)
    assert a.expanded <= u.expanded


def test_heuristic_never_beats_greedy_and_steps_are_consistent():
    rng = random.Random(1)
    picks = rng.sample(RIDE_IDS, 8)
    profile = random_profile(rng, picks)
    a = plan_astar(picks, 9 * 60, profile)
    for rule in ("nearest", "shortest-wait"):
        assert a.finish <= plan_greedy(picks, 9 * 60, profile, rule).finish + 1e-9
    assert sorted(a.order) == sorted(picks)
    t = 9 * 60
    for s in a.steps:
        assert s.arrive == pytest.approx(t + s.walk_minutes, abs=0.06)
        assert s.join_line >= s.arrive and s.board >= s.join_line
        t = s.ride_end


def test_closed_ride_is_infeasible():
    a, b = RIDE_IDS[:2]
    profile = constant_profile({a: 10, b: math.inf})
    with pytest.raises(Infeasible):
        plan_astar([a, b], 600, profile)


def test_resting_beats_joining_a_long_line():
    # Line is 90 min now, 5 min from 10:00. Best plan rests until 10:00.
    rid = RIDE_IDS[0]
    slots = [90.0] * 96
    for i in range(40, 96):
        slots[i] = 5.0
    p = plan_astar([rid], 9 * 60 + 30 - walk_minutes(START, rid), WaitProfile({rid: slots}, 15))
    s = p.steps[0]
    assert s.join_line == 600 and s.board == 605


def test_cross_park_walk_is_longer():
    rides = load_rides()
    dl = [r.id for r in rides.values() if r.park_id == 16]
    dca = [r.id for r in rides.values() if r.park_id == 17]
    assert walk_minutes(dl[0], dca[0]) > 5
    assert walk_minutes(dl[0], dl[1]) == pytest.approx(walk_minutes(dl[1], dl[0]))


def test_plan_can_start_at_a_ride():
    rng = random.Random(7)
    here, *picks = rng.sample(RIDE_IDS, 5)
    profile = random_profile(rng, picks)
    p = plan_astar(picks + [here], 600, profile, start_node=here)
    assert sorted(p.order) == sorted(picks)          # the ride you're at is not repeated
    assert p.steps[0].walk_from == here
    assert p.finish == pytest.approx(plan_brute_force(picks, 600, profile, start_node=here).finish)
