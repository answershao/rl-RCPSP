"""RCMPSP instance parsing, serial schedule generation, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import random

import numpy as np


ActivityId = tuple[int, int]


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


def parse_rcmp(path: str | Path) -> Instance:
    """Parse an MPSPLIB RCMP file into activities with integer demands."""
    path = Path(path)
    lines = [line.split() for line in path.read_text().splitlines() if line.split()]
    cursor = 0

    project_count = int(lines[cursor][0])
    cursor += 1
    resource_count = int(lines[cursor][0])
    cursor += 1
    capacities = tuple(map(int, lines[cursor]))
    cursor += 1
    if len(capacities) != resource_count:
        raise ValueError(f"{path}: expected {resource_count} resource capacities")

    activities: dict[ActivityId, Activity] = {}
    predecessors: dict[ActivityId, list[ActivityId]] = {}

    for project in range(1, project_count + 1):
        activity_count = int(lines[cursor][0])
        cursor += 1
        # The following availability vector identifies usable resource columns for
        # this project. Demands already encode unavailable resources as zero.
        availability = tuple(map(int, lines[cursor]))
        cursor += 1
        if len(availability) != resource_count:
            raise ValueError(f"{path}: project {project} has invalid availability vector")

        for activity_no in range(1, activity_count + 1):
            row = lines[cursor]
            cursor += 1
            minimum_columns = 1 + resource_count + 1
            if len(row) < minimum_columns:
                raise ValueError(f"{path}: truncated activity row for {project}:{activity_no}")

            duration = int(row[0])
            demand = tuple(map(int, row[1 : 1 + resource_count]))
            successor_count = int(row[1 + resource_count])
            successor_tokens = row[2 + resource_count :]
            if len(successor_tokens) != successor_count:
                raise ValueError(f"{path}: invalid successor count for {project}:{activity_no}")
            if any(value < 0 for value in demand) or duration < 0:
                raise ValueError(f"{path}: negative duration or demand for {project}:{activity_no}")

            successors = tuple(tuple(map(int, token.split(":"))) for token in successor_tokens)
            activity_id = (project, activity_no)
            activities[activity_id] = Activity(activity_id, duration, demand, successors)
            predecessors.setdefault(activity_id, [])
            for successor in successors:
                predecessors.setdefault(successor, []).append(activity_id)

    if cursor != len(lines):
        raise ValueError(f"{path}: unexpected trailing content")
    if any(successor not in activities for activity in activities.values() for successor in activity.successors):
        raise ValueError(f"{path}: successor references an unknown activity")
    if any(any(demand > capacity for demand, capacity in zip(activity.demand, capacities)) for activity in activities.values()):
        raise ValueError(f"{path}: an activity demand exceeds a resource capacity")

    return Instance(
        name=path.stem,
        capacities=capacities,
        activities=activities,
        predecessors={key: tuple(value) for key, value in predecessors.items()},
    )


def priority_fifo(activity: Activity) -> tuple[int, int, int]:
    return (activity.id[0], activity.id[1], 0)


def priority_shortest_duration(activity: Activity) -> tuple[int, int, int]:
    return (activity.duration, activity.id[0], activity.id[1])


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
