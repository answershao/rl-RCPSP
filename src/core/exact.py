"""CP-SAT exact and bounded-time solution support for RCMPSP instances."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.rcmpsp import Instance, Schedule, validate_schedule


@dataclass(frozen=True)
class ExactResult:
    """A CP-SAT scheduling result, including its proof status and bound."""

    schedule: Schedule
    status: str
    best_bound: float
    wall_time: float

def solve_exact(instance: Instance, *, time_limit: float = 60.0, workers: int = 1) -> ExactResult:
    """Minimize makespan with integer precedence and renewable-resource constraints.

    ``OPTIMAL`` means CP-SAT proved the returned schedule globally optimal.
    ``FEASIBLE`` means the time limit found a valid incumbent but did not prove
    that no shorter schedule exists.
    """
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError(
            "exact RCMPSP solving requires OR-Tools; install it with "
            "`python -m pip install 'ortools>=9.10,<10'`"
        ) from exc

    horizon = sum(activity.duration for activity in instance.activities.values())
    model = cp_model.CpModel()
    starts = {}
    ends = {}
    intervals = {}
    for activity_id, activity in instance.activities.items():
        start = model.NewIntVar(0, horizon, f"start_{activity_id[0]}_{activity_id[1]}")
        end = model.NewIntVar(0, horizon, f"end_{activity_id[0]}_{activity_id[1]}")
        starts[activity_id] = start
        ends[activity_id] = end
        intervals[activity_id] = model.NewIntervalVar(
            start, activity.duration, end, f"activity_{activity_id[0]}_{activity_id[1]}"
        )

    for activity_id, predecessors in instance.predecessors.items():
        for predecessor in predecessors:
            model.Add(ends[predecessor] <= starts[activity_id])
    for resource, capacity in enumerate(instance.capacities):
        resource_intervals = []
        demands = []
        for activity_id, activity in instance.activities.items():
            if activity.demand[resource]:
                resource_intervals.append(intervals[activity_id])
                demands.append(activity.demand[resource])
        if resource_intervals:
            model.AddCumulative(resource_intervals, demands, capacity)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, list(ends.values()))
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    status_code = solver.Solve(model)
    status = solver.StatusName(status_code)
    if status not in {"OPTIMAL", "FEASIBLE"}:
        raise RuntimeError(f"CP-SAT could not produce a schedule: {status}")
    schedule = Schedule(
        starts={activity_id: solver.Value(start) for activity_id, start in starts.items()},
        finishes={activity_id: solver.Value(end) for activity_id, end in ends.items()},
        makespan=solver.Value(makespan),
    )
    validate_schedule(instance, schedule)
    return ExactResult(schedule, status, solver.BestObjectiveBound(), solver.WallTime())
