"""Deterministic single-pass priority-rule baselines for single-project RCPSP.

Static rules compute CPM earliest/latest times and graph measures once per
instance and expose each rule as an ``{activity_id: score}`` map (higher score
= scheduled earlier; ties broken by ascending activity id), which is consumed
by the serial SGS (:func:`src.core.rcpsp.generate_schedule`) or the parallel
SGS (:func:`src.core.rcpsp.generate_schedule_parallel`).

WCS (worst case slack, Kolisch 1996) is dynamic and is implemented inside the
parallel SGS only (``generate_schedule_parallel(..., wcs=True)``), matching the
original publication.  ``Random`` reuses the seeded per-activity random scores
from :func:`src.core.rcpsp.random_priorities`.
"""

from __future__ import annotations

from src.core.rcpsp import (
    ActivityId,
    Instance,
    generate_schedule,
    generate_schedule_parallel,
    latest_start_times,
    random_priorities,
)

# Rules with a fixed per-activity measure (min or max variant is listed).
STATIC_RULES = ("FIFO", "SPT", "LPT", "EST", "EFT", "LST", "LFT", "MSLK", "MTS", "GRPW", "GRD")

# Measures that are *minimised* (time-window and id-based rules).
_MINIMIZE_RULES = {"FIFO", "SPT", "EST", "EFT", "LST", "LFT", "MSLK"}


def _topological_order(instance: Instance) -> list[ActivityId]:
    indegree = {aid: len(instance.predecessors[aid]) for aid in instance.activities}
    queue = [aid for aid, degree in indegree.items() if degree == 0]
    order: list[ActivityId] = []
    while queue:
        aid = queue.pop()
        order.append(aid)
        for succ in instance.activities[aid].successors:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)
    if len(order) != len(instance.activities):
        raise ValueError("precedence graph is cyclic")
    return order


def _cpm_times(instance: Instance, horizon: int | None = None) -> dict[ActivityId, tuple[int, int]]:
    """Earliest (est/eft) and latest (lst/lft) times ignoring resources."""
    order = _topological_order(instance)
    durations = {aid: instance.activities[aid].duration for aid in instance.activities}
    est: dict[ActivityId, int] = {}
    eft: dict[ActivityId, int] = {}
    for aid in order:
        ready = max((eft[pred] for pred in instance.predecessors[aid]), default=0)
        est[aid] = ready
        eft[aid] = ready + durations[aid]
    latest = latest_start_times(instance, horizon=horizon)
    return {aid: (est[aid], eft[aid], latest[aid], latest[aid] + durations[aid]) for aid in instance.activities}


def _successor_statistics(instance: Instance) -> dict[ActivityId, tuple[int, int]]:
    """For every activity: (number of total successors, total successor duration)."""
    order = _topological_order(instance)
    durations = {aid: instance.activities[aid].duration for aid in instance.activities}
    successors: dict[ActivityId, set[ActivityId]] = {aid: set() for aid in instance.activities}
    for aid in reversed(order):
        own: set[ActivityId] = set()
        for succ in instance.activities[aid].successors:
            own.add(succ)
            own |= successors[succ]
        successors[aid] = own
    return {
        aid: (len(successors[aid]), sum(durations[succ] for succ in successors[aid]))
        for aid in instance.activities
    }


def _raw_measure(instance: Instance, rule: str, horizon: int | None = None) -> dict[ActivityId, float]:
    """Lower-is-better raw measure for a static rule (FIFO uses the file order)."""
    order = _topological_order(instance)
    index = {aid: i for i, aid in enumerate(order)}
    durations = {aid: instance.activities[aid].duration for aid in instance.activities}
    if rule == "FIFO":
        return {aid: float(index[aid]) for aid in instance.activities}
    if rule == "SPT":
        return {aid: float(durations[aid]) for aid in instance.activities}
    if rule == "LPT":
        return {aid: float(-durations[aid]) for aid in instance.activities}
    times = _cpm_times(instance, horizon=horizon)
    if rule in ("EST", "EFT", "LST", "LFT", "MSLK"):
        if rule == "MSLK":
            return {aid: float(times[aid][2] - times[aid][0]) for aid in instance.activities}
        slot = {"EST": 0, "EFT": 1, "LST": 2, "LFT": 3}[rule]
        return {aid: float(times[aid][slot]) for aid in instance.activities}
    stats = _successor_statistics(instance)
    if rule == "MTS":
        return {aid: float(-stats[aid][0]) for aid in instance.activities}
    if rule == "GRPW":
        return {aid: float(-(durations[aid] + stats[aid][1])) for aid in instance.activities}
    if rule == "GRD":
        return {
            aid: float(-(durations[aid] * sum(instance.activities[aid].demand)))
            for aid in instance.activities
        }
    raise ValueError(
        f"unknown static rule: {rule!r}; choose from {sorted(STATIC_RULES)}"
    )


def static_priorities(
    instance: Instance, rule: str, *, horizon: int | None = None
) -> dict[ActivityId, float]:
    """Score map (higher = earlier) for one static rule on one instance."""
    if rule not in STATIC_RULES:
        raise ValueError(f"unknown static rule: {rule!r}; choose from {STATIC_RULES}")
    raw = _raw_measure(instance, rule, horizon=horizon)
    if rule in _MINIMIZE_RULES:
        return {aid: -value for aid, value in raw.items()}
    return raw


# Features used by GPHH as terminal values.  ``dur`` is the duration; ``slack``
# is LST - EST; ``nsucc``/``sdur`` are total (transitive) successor count and
# duration; ``rdem`` is the summed renewable-resource demand.
FEATURE_NAMES = ("dur", "est", "eft", "lst", "lft", "slack", "nsucc", "sdur", "rdem")


def activity_features(instance: Instance) -> dict[ActivityId, dict[str, float]]:
    """Static per-activity features for one instance (raw, unscaled).

    Shares the exact CPM time windows and transitive successor statistics used
    by :func:`static_priorities`, so GPHH terminals are comparable with every
    static rule.  ``nsucc``/``sdur`` count transitive successors, matching the
    MTS/GRPW definitions.  Features are raw floats; callers may scale them.
    """
    times = _cpm_times(instance)
    stats = _successor_statistics(instance)
    durations = {aid: instance.activities[aid].duration for aid in instance.activities}
    return {
        aid: {
            "dur": float(durations[aid]),
            "est": float(times[aid][0]),
            "eft": float(times[aid][1]),
            "lst": float(times[aid][2]),
            "lft": float(times[aid][3]),
            "slack": float(times[aid][2] - times[aid][0]),
            "nsucc": float(stats[aid][0]),
            "sdur": float(stats[aid][1]),
            "rdem": float(sum(instance.activities[aid].demand)),
        }
        for aid in instance.activities
    }


def rule_makespan(
    instance: Instance,
    rule: str,
    *,
    scheme: str = "serial",
    seed: int = 17,
    horizon: int | None = None,
) -> int:
    """Single-pass makespan of ``rule`` under ``scheme`` in {"serial", "parallel"}."""
    if rule == "Random":
        priorities = random_priorities(instance, seed)
    elif rule == "WCS":
        if scheme != "parallel":
            raise ValueError("WCS is defined for the parallel SGS only")
        priorities = {}
    else:
        priorities = static_priorities(instance, rule, horizon=horizon)
    if scheme == "serial":
        if rule == "WCS":
            raise ValueError("WCS is defined for the parallel SGS only")
        schedule = generate_schedule(instance, priorities)
    elif scheme == "parallel":
        schedule = (
            generate_schedule_parallel(instance, wcs=True)
            if rule == "WCS"
            else generate_schedule_parallel(instance, priorities)
        )
    else:
        raise ValueError(f"unknown scheme: {scheme!r}; choose from {'serial', 'parallel'}")
    return schedule.makespan


# Column naming used by the baseline matrix.  WCS exists only under ``parallel``.
def column_name(scheme: str, rule: str) -> str:
    return f"{scheme}_{rule}"


def available_columns(schemes: tuple[str, ...] = ("serial", "parallel")) -> list[str]:
    columns: list[str] = []
    for scheme in schemes:
        rules = STATIC_RULES + ("Random",)
        if scheme == "parallel":
            rules = rules + ("WCS",)
        columns.extend(column_name(scheme, rule) for rule in rules)
    return columns


def all_makespans(
    instance: Instance,
    *,
    seed: int = 17,
    schemes: tuple[str, ...] = ("serial", "parallel"),
) -> dict[str, int]:
    """Run every rule x scheme combination on one instance."""
    result: dict[str, int] = {}
    for scheme in schemes:
        rules = list(STATIC_RULES) + ["Random"]
        if scheme == "parallel":
            rules.append("WCS")
        for rule in rules:
            result[column_name(scheme, rule)] = rule_makespan(
                instance, rule, scheme=scheme, seed=seed
            )
    return result
