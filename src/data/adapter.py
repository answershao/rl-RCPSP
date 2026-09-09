"""Bridge between the parser data layer and the core scheduling kernel.

The parser in ``src.data.parsers`` (PSPLIB ``.sm`` / ProGen ``.rcp`` /
Patterson) produces ``RCPSPInstance`` with 0-indexed flat activity ids.  The
scheduling kernel (``src.core.rcpsp.Instance``) uses the same 0-indexed
activity ids and additionally materialises the predecessor map.

All benchmark suites (PSPLIB j30-j120, RG30, RG300, Patterson) are
single-project instances, so the adapter keeps the raw file ordering
unchanged, including any dummy source/sink rows already present in
``.sm``/``.rcp`` (their durations are 0, so makespans are unaffected).
"""

from __future__ import annotations

from pathlib import Path

from src.core.rcpsp import Activity, ActivityId, Instance
from src.data.parsers import RCPSPInstance, load_instance


def to_core_instance(inst: RCPSPInstance, *, name: str | None = None) -> Instance:
    """Convert one parsed single-project instance to the kernel ``Instance`` shape.

    ``name`` overrides the parsed file-stem name.  The PPO static graph cache
    requires unique instance names, and file stems collide across RG30 ``Set``
    directories, so callers training on ``splits.json`` should pass a unique
    identifier (e.g. the data-root-relative path without its extension).
    """
    activities: dict[ActivityId, Activity] = {}
    predecessors: dict[ActivityId, list[ActivityId]] = {}

    for raw in range(inst.n_activities):
        successors = tuple(int(succ) for succ in inst.successors[raw])
        demand = tuple(int(value) for value in inst.demands[raw])
        if any(value < 0 for value in demand) or int(inst.durations[raw]) < 0:
            raise ValueError(f"{inst.name}: negative duration or demand for activity {raw}")
        activities[raw] = Activity(
            id=raw,
            duration=int(inst.durations[raw]),
            demand=demand,
            successors=successors,
        )
        predecessors.setdefault(raw, [])
        for succ in successors:
            predecessors.setdefault(succ, []).append(raw)

    capacities = tuple(int(value) for value in inst.capacities)
    if len(capacities) != inst.n_renewable:
        raise ValueError(f"{inst.name}: capacity/activity resource count mismatch")
    if any(
        any(demand > capacity for demand, capacity in zip(activity.demand, capacities))
        for activity in activities.values()
    ):
        raise ValueError(f"{inst.name}: an activity demand exceeds a resource capacity")

    return Instance(
        name=name if name is not None else inst.name,
        capacities=capacities,
        activities=activities,
        predecessors={key: tuple(value) for key, value in predecessors.items()},
    )


def load_core_instance(path: str | Path, *, name: str | None = None) -> Instance:
    """Parse one single-project ``.sm``/``.rcp`` file and adapt it.

    This is the canonical file-path entry point for the RL stack: every
    environment / evaluation entry point that loads an instance from disk goes
    through ``src.data.parsers`` + this adapter, so PPO shares the exact data
    representation with the baselines.
    """
    parsed = load_instance(path)
    return to_core_instance(parsed, name=name if name is not None else Path(path).stem)
