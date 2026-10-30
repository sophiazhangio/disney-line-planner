"""Find the fastest order to ride a set of attractions.

State: (where I am, which rides are left, clock time).
Action: walk to a ride that is left, (optionally rest until the line is
shorter), wait in line, ride it.
Goal: no rides left. Cost: time from the start until the last ride ends.

This is A* search. Two properties make it exact:

1. FIFO travel times. `WaitProfile.best_boarding` lets you rest before joining
   a line, so reaching a ride earlier never makes you board later. Then, for a
   given (location, rides left), the earliest clock time dominates every later
   one, and we only need to keep the earliest.
2. An admissible heuristic. Every ride that is left still costs at least
   its ride length + its smallest wait from now on + the shortest walk into it
   from any place it could be entered from (where I am, or another ride that is
   left). Each ride is entered exactly once, so the sum never overestimates.

`plan_ucs` is the same search with h = 0 (uniform-cost search), kept so the
README can compare how many states each one expands.
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

from .park import START, load_rides, walk_matrix
from .waits import WaitProfile


@dataclass
class Step:
    ride_id: int
    ride: str
    walk_from: int
    walk_minutes: float
    arrive: float
    join_line: float
    wait_minutes: float
    board: float
    ride_end: float


@dataclass
class Plan:
    order: list[int]
    steps: list[Step]
    start: float
    finish: float
    expanded: int = 0
    algorithm: str = "astar"
    notes: list[str] = field(default_factory=list)

    @property
    def total_minutes(self) -> float:
        return self.finish - self.start

    @property
    def total_wait(self) -> float:
        return sum(s.wait_minutes for s in self.steps)

    @property
    def total_walk(self) -> float:
        return sum(s.walk_minutes for s in self.steps)

    @property
    def total_rest(self) -> float:
        return sum(s.join_line - s.arrive for s in self.steps)


class Infeasible(Exception):
    pass


def _simulate(order: list[int], start: float, profile: WaitProfile,
              walk: dict[tuple[int, int], float], start_node: int = START) -> list[Step] | None:
    rides = load_rides()
    loc, t, steps = start_node, start, []
    for rid in order:
        w = walk[(loc, rid)]
        arrive = t + w
        join, board = profile.best_boarding(rid, arrive)
        if math.isinf(board):
            return None
        end = board + rides[rid].ride_minutes
        steps.append(Step(rid, rides[rid].name, loc, round(w, 1), arrive, join, board - join, board, end))
        loc, t = rid, end
    return steps


def _make_plan(order, start, profile, walk, expanded, algorithm, start_node=START) -> Plan:
    steps = _simulate(order, start, profile, walk, start_node)
    assert steps is not None
    return Plan(order, steps, start, steps[-1].ride_end if steps else start, expanded, algorithm)


def plan_astar(ride_ids: list[int], start: float, profile: WaitProfile,
               use_heuristic: bool = True, start_node: int = START) -> Plan:
    """start_node: START (the Esplanade) or the id of the ride you are standing at."""
    rides = load_rides()
    ride_ids = [r for r in dict.fromkeys(ride_ids) if r != start_node]
    unknown = [r for r in ride_ids + [start_node] if r != START and r not in rides]
    if unknown:
        raise ValueError(f"unknown ride ids: {unknown}")
    n = len(ride_ids)
    if n == 0:
        return Plan([], [], start, start, 0)
    nodes = [start_node] + ride_ids
    walk = walk_matrix(nodes)
    bit = {rid: 1 << i for i, rid in enumerate(ride_ids)}
    full = (1 << n) - 1
    dur = {rid: rides[rid].ride_minutes for rid in ride_ids}

    def members(mask: int):
        return (rid for rid in ride_ids if mask & bit[rid])

    def h(loc: int, mask: int, t: float) -> float:
        if not use_heuristic:
            return 0.0
        total = 0.0
        for r in members(mask):
            mw = profile.min_wait_from(r, t)
            if math.isinf(mw):
                return math.inf  # this ride can no longer be done today
            entry = walk[(loc, r)]
            for s in members(mask):
                if s != r and walk[(s, r)] < entry:
                    entry = walk[(s, r)]
            total += dur[r] + mw + entry
        return total

    best_t: dict[tuple[int, int], float] = {(start_node, full): start}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(start_node, full): None}
    tie = itertools.count()
    frontier = [(start + h(start_node, full, start), next(tie), start, start_node, full)]
    expanded = 0
    while frontier:
        _, _, t, loc, mask = heapq.heappop(frontier)
        if t > best_t[(loc, mask)]:
            continue  # stale entry: we already reached this state earlier
        if mask == 0:
            order, key = [], (loc, mask)
            while parent[key] is not None:
                order.append(key[0])
                key = parent[key]
            return _make_plan(order[::-1], start, profile, walk, expanded,
                              "astar" if use_heuristic else "ucs", start_node)
        expanded += 1
        for r in members(mask):
            _, board = profile.best_boarding(r, t + walk[(loc, r)])
            if math.isinf(board):
                continue
            t2, key = board + dur[r], (r, mask & ~bit[r])
            if t2 < best_t.get(key, math.inf):
                f = t2 + h(r, key[1], t2)
                if math.isinf(f):
                    continue
                best_t[key], parent[key] = t2, (loc, mask)
                heapq.heappush(frontier, (f, next(tie), t2, r, key[1]))
    raise Infeasible("no order fits before closing time (a ride may be closed)")


def plan_ucs(ride_ids: list[int], start: float, profile: WaitProfile, start_node: int = START) -> Plan:
    return plan_astar(ride_ids, start, profile, use_heuristic=False, start_node=start_node)


def plan_brute_force(ride_ids: list[int], start: float, profile: WaitProfile, start_node: int = START) -> Plan:
    """Try every order. Only for checking the search on small inputs."""
    ride_ids = [r for r in ride_ids if r != start_node]
    walk = walk_matrix([start_node] + ride_ids)
    best, count = None, 0
    for order in itertools.permutations(ride_ids):
        count += 1
        steps = _simulate(list(order), start, profile, walk, start_node)
        if steps and (best is None or steps[-1].ride_end < best[1][-1].ride_end):
            best = (list(order), steps)
    if best is None:
        raise Infeasible("no feasible order")
    return Plan(best[0], best[1], start, best[1][-1].ride_end, count, "brute-force")


def plan_greedy(ride_ids: list[int], start: float, profile: WaitProfile, rule: str = "nearest",
                start_node: int = START) -> Plan:
    """What people usually do: go to the closest ride next ("nearest") or the
    ride with the shortest line right now ("shortest-wait"). Used as a baseline."""
    ride_ids = [r for r in ride_ids if r != start_node]
    walk = walk_matrix([start_node] + ride_ids)
    rides = load_rides()
    loc, t, left, order = start_node, start, list(ride_ids), []
    while left:
        def score(r):
            if rule == "nearest":
                return walk[(loc, r)]
            return profile.wait(r, t + walk[(loc, r)])
        r = min(left, key=score)
        _, board = profile.best_boarding(r, t + walk[(loc, r)])
        if math.isinf(board):
            raise Infeasible(f"{rides[r].name} is closed or the line closes before you get there")
        order.append(r)
        left.remove(r)
        loc, t = r, board + rides[r].ride_minutes
    return _make_plan(order, start, profile, walk, 0, f"greedy-{rule}", start_node)
