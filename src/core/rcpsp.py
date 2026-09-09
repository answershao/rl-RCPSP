"""Core RCPSP instance types, schedule generation, and validation.

Single-project RCPSP instances are parsed by ``src.data.parsers`` and adapted
by ``src.data.adapter``; this module owns the DAG ``Instance`` representation,
the serial/parallel schedule-generation schemes (SGS) and schedule validation
used by every baseline and by the RL environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import random

import numpy as np


# A single-project activity is identified by its 0-indexed position in the
# source file (activity 0 = dummy source, n-1 = dummy sink).
ActivityId = int


@dataclass(frozen=True)
class Activity:
    id: ActivityId
    duration: int
    demand: tuple[int, ...]
    successors: tuple[ActivityId, ...]


@dataclass(frozen=True)
class Instance:
    name: str
    capacities: tuple[int, ...]
    activities: dict[ActivityId, Activity]
    predecessors: dict[ActivityId, tuple[ActivityId, ...]]

    @property
    def resource_count(self) -> int:
        return len(self.capacities)


@dataclass(frozen=True)
class Schedule:
    starts: dict[ActivityId, int]
    finishes: dict[ActivityId, int]
    makespan: int


def priority_fifo(activity: Activity) -> int:
    return activity.id


def priority_shortest_duration(activity: Activity) -> tuple[int, int]:
    return (activity.duration, activity.id)


def random_priorities(instance: Instance, seed: int) -> dict[ActivityId, float]:
    rng = random.Random(seed)
    return {activity_id: rng.random() for activity_id in instance.activities}


def baseline_makespans(instance: Instance, seed: int) -> dict[str, int]:
    """Evaluate the deterministic scheduling baselines for one instance."""
    return {
        "fifo": generate_schedule(instance, priority_fifo).makespan,
        "shortest": generate_schedule(instance, priority_shortest_duration).makespan,
        "random": generate_schedule(instance, random_priorities(instance, seed)).makespan,
    }


def latest_start_times(instance: Instance, horizon: int | None = None) -> dict[ActivityId, int]:
    """Backward-pass latest start times.

    ``horizon`` defaults to the total duration sum (the same safe upper bound
    the environment uses).  For every activity, LFT(a) = min over successors s
    of LST(s), and LST(a) = LFT(a) - duration(a).  A sink activity therefore
    gets LST = horizon.  Recursion is memoised and terminates because the
    precedence graph is a DAG.
    """
    if horizon is None:
        horizon = sum(activity.duration for activity in instance.activities.values())

    lft_cache: dict[ActivityId, int] = {}

    def latest_start(activity_id: ActivityId) -> int:
        activity = instance.activities[activity_id]
        return latest_finish(activity_id) - activity.duration

    def latest_finish(activity_id: ActivityId) -> int:
        cached = lft_cache.get(activity_id)
        if cached is not None:
            return cached
        activity = instance.activities[activity_id]
        bound = min((latest_start(succ) for succ in activity.successors), default=horizon)
        lft_cache[activity_id] = bound
        return bound

    return {activity_id: latest_start(activity_id) for activity_id in instance.activities}


def priority_latest_start(instance: Instance) -> Callable[[Activity], object]:
    """Priority rule LST: schedule the eligible activity with the smallest
    latest start time first; ties break by activity id for determinism."""
    latest = latest_start_times(instance)
    return lambda activity: (latest[activity.id], activity.id)


def generate_schedule(
    instance: Instance,
    priority: Callable[[Activity], object] | dict[ActivityId, float] = priority_fifo,
) -> Schedule:
    """Build a feasible schedule with a serial SGS and integer resource profiles.

    The priority callback or score map chooses among precedence-eligible
    activities. Resource allocation remains the instance's fixed integer
    demand throughout a non-preemptive activity.
    """
    unscheduled = set(instance.activities)
    starts: dict[ActivityId, int] = {}
    finishes: dict[ActivityId, int] = {}
    usage: list[list[int]] = []

    def rank(activity: Activity) -> object:
        if isinstance(priority, dict):
            return (-priority[activity.id], activity.id)
        return priority(activity)

    while unscheduled:
        eligible = [
            activity_id
            for activity_id in unscheduled
            if all(predecessor in finishes for predecessor in instance.predecessors[activity_id])
        ]
        if not eligible:
            raise ValueError("precedence graph is cyclic or has a missing predecessor")

        activity_id = min(eligible, key=lambda item: rank(instance.activities[item]))
        serial_sgs_insert(instance, activity_id, starts, finishes, usage)
        unscheduled.remove(activity_id)

    schedule = Schedule(starts=starts, finishes=finishes, makespan=max(finishes.values(), default=0))
    validate_schedule(instance, schedule)
    return schedule


def generate_schedule_parallel(
    instance: Instance,
    priority: Callable[[Activity], object] | dict[ActivityId, float] = priority_fifo,
    *,
    wcs: bool = False,
) -> Schedule:
    """Build a feasible schedule with the parallel (time-increment) SGS.

    At every schedule time ``t`` the precedence-eligible activities are scanned
    in priority order and every activity that fits into the remaining capacity
    is started at ``t``.  Time then advances to the next activity finish where
    capacity/eligibility may change.  ``priority`` follows the same conventions
    as :func:`generate_schedule` (callable sort key or ``{id: score}`` map with
    higher scores scheduled first; ties broken by ascending activity id).

    With ``wcs=True`` the Kolisch (1996) worst-case-slack rule is used instead
    of a static priority.  Among the activities that are precedence-eligible
    and resource-feasible at ``t``, the next one started minimises

        WCS(j) = LST(j) - max_{i != j in the decision set} E(i, j),

    where E(i, j) is the earliest feasible start time of ``j`` once ``i`` is
    started at ``t``.  The rule prefers the activity that would be delayed past
    its latest start the most if any other eligible activity were started now.
    """
    if wcs:
        latest = latest_start_times(instance)
    ids = tuple(sorted(instance.activities))
    durations = {aid: instance.activities[aid].duration for aid in ids}
    demands = {aid: instance.activities[aid].demand for aid in ids}
    capacities = np.asarray(instance.capacities, dtype=np.int32)
    horizon = sum(durations.values())
    usage = np.zeros((horizon + 1, instance.resource_count), dtype=np.int32)
    starts: dict[ActivityId, int] = {}
    finishes: dict[ActivityId, int] = {}

    def feasible_at(activity_id: ActivityId, t: int) -> bool:
        duration = durations[activity_id]
        if duration == 0:
            return True
        limit = capacities - np.asarray(demands[activity_id], dtype=np.int32)
        return bool(np.all(usage[t : t + duration] <= limit))

    def reserve(activity_id: ActivityId, t: int) -> None:
        duration = durations[activity_id]
        starts[activity_id] = t
        finishes[activity_id] = t + duration
        if duration:
            usage[t : t + duration] += np.asarray(demands[activity_id], dtype=np.int32)

    def rank(activity: Activity) -> object:
        if isinstance(priority, dict):
            return (-priority[activity.id], activity.id)
        return priority(activity)

    t = 0
    while len(finishes) < len(ids):
        changed = True
        while changed:
            changed = False
            eligible = [
                aid
                for aid in ids
                if aid not in finishes
                and all(
                    pred in finishes and finishes[pred] <= t
                    for pred in instance.predecessors[aid]
                )
            ]
            if not eligible:
                break
            if wcs:
                startable = [aid for aid in eligible if feasible_at(aid, t)]
                if not startable:
                    break
                best: ActivityId | None = None
                best_key: tuple[float, ActivityId] | None = None
                for candidate in startable:
                    if len(startable) == 1:
                        worst_delay = t
                    else:
                        worst_delay = -1
                        for other in startable:
                            if other == candidate:
                                continue
                            if durations[other] == 0:
                                probe = usage
                            else:
                                probe = usage.copy()
                                probe[t : t + durations[other]] += np.asarray(
                                    demands[other], dtype=np.int32
                                )
                            earliest = _earliest_feasible_start(
                                probe,
                                capacities=instance.capacities,
                                demand=demands[candidate],
                                duration=durations[candidate],
                                earliest=t,
                            )
                            if earliest > worst_delay:
                                worst_delay = earliest
                    key = (float(latest[candidate] - worst_delay), candidate)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = candidate
                if best is not None:
                    reserve(best, t)
                    changed = True
            else:
                ordered = sorted(eligible, key=lambda aid: rank(instance.activities[aid]))
                for aid in ordered:
                    if aid in finishes:
                        continue
                    if feasible_at(aid, t):
                        reserve(aid, t)
                        changed = True
        next_t = min((finish for finish in finishes.values() if finish > t), default=None)
        if next_t is None:
            if len(finishes) < len(ids):
                raise RuntimeError("parallel SGS stalled before scheduling every activity")
            break
        t = next_t

    schedule = Schedule(starts=starts, finishes=finishes, makespan=max(finishes.values(), default=0))
    validate_schedule(instance, schedule)
    return schedule


def serial_sgs_insert(
    instance: Instance,
    activity_id: ActivityId,
    starts: dict[ActivityId, int],
    finishes: dict[ActivityId, int],
    usage: list[list[int]] | np.ndarray,
    *,
    capacities_array: np.ndarray | None = None,
    demand_array: np.ndarray | None = None,
    capacity_limit_array: np.ndarray | None = None,
    current_makespan: int | None = None,
) -> tuple[int, int]:
    """Insert one precedence-eligible activity at its earliest feasible start.

    ``starts``, ``finishes``, and ``usage`` are updated in place so callers can
    construct a schedule one priority decision at a time.
    """
    if activity_id in starts:
        raise ValueError(f"activity {activity_id} is already scheduled")
    missing = [pred for pred in instance.predecessors[activity_id] if pred not in finishes]
    if missing:
        raise ValueError(f"activity {activity_id} has unscheduled predecessors: {missing}")

    activity = instance.activities[activity_id]
    start, finish = preview_serial_sgs_insert(
        instance,
        activity_id,
        finishes,
        usage,
        capacities_array=capacities_array,
        demand_array=demand_array,
        capacity_limit_array=capacity_limit_array,
        current_makespan=current_makespan,
    )
    _reserve(
        usage,
        demand_array if demand_array is not None else activity.demand,
        start,
        finish,
    )
    starts[activity_id] = start
    finishes[activity_id] = finish
    return start, finish


def preview_serial_sgs_insert(
    instance: Instance,
    activity_id: ActivityId,
    finishes: dict[ActivityId, int],
    usage: list[list[int]] | np.ndarray,
    *,
    capacities_array: np.ndarray | None = None,
    demand_array: np.ndarray | None = None,
    capacity_limit_array: np.ndarray | None = None,
    current_makespan: int | None = None,
) -> tuple[int, int]:
    """Return an eligible activity's serial-SGS placement without mutation."""
    missing = [pred for pred in instance.predecessors[activity_id] if pred not in finishes]
    if missing:
        raise ValueError(f"activity {activity_id} has unscheduled predecessors: {missing}")

    activity = instance.activities[activity_id]
    ready = max((finishes[pred] for pred in instance.predecessors[activity_id]), default=0)
    start = _earliest_feasible_start(
        usage,
        capacities_array if capacities_array is not None else instance.capacities,
        demand_array if demand_array is not None else activity.demand,
        activity.duration,
        ready,
        capacity_limit_array=capacity_limit_array,
        current_makespan=current_makespan,
    )
    return start, start + activity.duration


def _earliest_feasible_start(
    usage: list[list[int]] | np.ndarray,
    capacities: tuple[int, ...],
    demand: tuple[int, ...],
    duration: int,
    earliest: int,
    *,
    capacity_limit_array: np.ndarray | None = None,
    current_makespan: int | None = None,
) -> int:
    if duration == 0:
        return earliest
    start = earliest
    if isinstance(usage, np.ndarray):
        capacities_array = np.asarray(capacities, dtype=np.int32)
        demand_array = np.asarray(demand, dtype=np.int32)
        capacity_limit_array = (
            capacities_array - demand_array
            if capacity_limit_array is None
            else capacity_limit_array
        )
        if current_makespan is not None and earliest >= current_makespan:
            return earliest
        search_stop = (
            len(usage)
            if current_makespan is None
            else min(len(usage), current_makespan + duration)
        )
        conflicts = np.any(
            usage[earliest:search_stop] > capacity_limit_array, axis=1
        )
        cumulative = np.cumsum(conflicts, dtype=np.int32)
        window_conflicts = cumulative[duration - 1:].copy()
        if duration < cumulative.size:
            window_conflicts[1:] -= cumulative[:-duration]
        feasible = np.flatnonzero(window_conflicts == 0)
        if feasible.size:
            return earliest + int(feasible[0])
        raise RuntimeError("serial SGS search exceeded the schedule horizon")
    while True:
        finish = start + duration
        if all(
            all((usage[time][resource] if time < len(usage) else 0) + demand[resource] <= capacities[resource]
                for resource in range(len(capacities)))
            for time in range(start, finish)
        ):
            return start
        start += 1


def _reserve(usage: list[list[int]] | np.ndarray, demand: tuple[int, ...], start: int, finish: int) -> None:
    if isinstance(usage, np.ndarray):
        usage[start:finish] += np.asarray(demand, dtype=np.int32)
        return
    while len(usage) < finish:
        usage.append([0] * len(demand))
    for time in range(start, finish):
        for resource, amount in enumerate(demand):
            usage[time][resource] += amount


def validate_schedule(instance: Instance, schedule: Schedule) -> None:
    """Raise ValueError if precedence, duration, or integer capacity constraints fail."""
    if set(schedule.starts) != set(instance.activities) or set(schedule.finishes) != set(instance.activities):
        raise ValueError("schedule does not contain every activity")
    if schedule.makespan != max(schedule.finishes.values(), default=0):
        raise ValueError("makespan does not match activity finish times")

    usage = [[0] * instance.resource_count for _ in range(schedule.makespan)]
    for activity_id, activity in instance.activities.items():
        start, finish = schedule.starts[activity_id], schedule.finishes[activity_id]
        if start < 0 or finish != start + activity.duration:
            raise ValueError(f"invalid timing for activity {activity_id}")
        for predecessor in instance.predecessors[activity_id]:
            if schedule.finishes[predecessor] > start:
                raise ValueError(f"precedence violation: {predecessor} -> {activity_id}")
        for time in range(start, finish):
            for resource, amount in enumerate(activity.demand):
                usage[time][resource] += amount

    for time, amounts in enumerate(usage):
        for resource, (amount, capacity) in enumerate(zip(amounts, instance.capacities)):
            if amount > capacity:
                raise ValueError(f"resource {resource} exceeds capacity at time {time}: {amount} > {capacity}")
