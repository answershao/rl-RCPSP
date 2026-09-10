"""ProGen-style RCPSP instance generator with explicit parameter control.

Why this exists
---------------
The PPO policy is trained on one instance pool and evaluated on another.  A
pool can only be defended if the *parameters that drive the policy's input
distribution* are set deliberately rather than inherited from whatever corpus
happened to be on disk.  This module generates single-project RCPSP instances
with direct control over the four standard ProGen/Kolisch-Sprecher knobs:

``n``   number of real activities (plus dummy source/sink)
``RF``  resource factor -- probability that an activity consumes a resource
``RS``  resource strength -- capacity slack, ``0`` = tightest, ``1`` = loose
``NC``  network complexity -- arcs per node, counted over ``n + 2`` nodes

The realised values match the requested ones by construction, so a generated
pool can be aimed at the same coordinates as a benchmark grid (see
``scripts/generate_pool.py``) and verified with ``scripts/instance_stats.py``.

Definitions follow Kolisch & Sprecher (1996) / the PSPLIB generator:

* durations ``p_i ~ U{1..max_duration}``
* demands ``d_ik = U{1..max_demand}`` with probability ``RF``, else ``0``
* capacity ``a_k = r_min_k + RS * (r_max_k - r_min_k)`` where
  ``r_min_k = max_i d_ik`` and ``r_max_k = sum_i d_ik``

The precedence network is generated level by level so that the critical-path
depth is controllable (PSPLIB networks are wide and shallow: ``j30`` has a
critical path of ~41 duration units, i.e. roughly 7-8 activities deep).

Everything is driven by a single integer seed, so a pool is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.parsers import RCPSPInstance


# Calibrated against PSPLIB's realised critical paths (j30 ~41, j60 ~72, j90 ~87,
# j120 ~111 duration units at a mean duration of 5.5, i.e. ~7.5 / 13 / 16 / 20
# activities deep).  The padding arcs lengthen the longest path beyond the level
# count, so the coefficients below are fitted to the measured CP, not to depth.
def depth_for(n_activities: int) -> int:
    """Number of precedence levels whose critical path matches PSPLIB's depth."""
    return max(2, int(round(0.122 * n_activities + 1.4)))


@dataclass(frozen=True)
class GeneratorSpec:
    """One point in generator-parameter space."""

    n_activities: int
    resource_factor: float
    resource_strength: float
    network_complexity: float
    n_resources: int = 4
    max_duration: int = 10
    max_demand: int = 10
    depth: int | None = None

    def __post_init__(self) -> None:
        if self.n_activities < 2:
            raise ValueError("n_activities must be at least 2")
        if self.n_resources < 1:
            raise ValueError("n_resources must be positive")
        if not 0.0 < self.resource_factor <= 1.0:
            raise ValueError("resource_factor must be in (0, 1]")
        if not 0.0 <= self.resource_strength <= 1.0:
            raise ValueError("resource_strength must be in [0, 1]")
        if self.network_complexity <= 0.0:
            raise ValueError("network_complexity must be positive")

    @property
    def levels(self) -> int:
        return self.depth if self.depth is not None else depth_for(self.n_activities)

    def label(self) -> str:
        return (
            f"n{self.n_activities}_rf{self.resource_factor:g}"
            f"_rs{self.resource_strength:g}_nc{self.network_complexity:g}"
        )


def _build_precedence(spec: GeneratorSpec, rng: np.random.Generator) -> list[list[int]]:
    """Return 0-indexed successor lists for ``n + 2`` nodes (0 = source).

    The network is grown level by level, which makes acyclicity a property of
    the construction rather than something to check afterwards:

    1. every real activity gets exactly one predecessor, drawn from the level
       below (level 1 draws the dummy source) -- this is the tree skeleton
       PSPLIB uses, so the arc count starts at ``n`` rather than at ``2n``;
    2. every activity left without a successor is attached to a random activity
       one level up, or to the dummy sink at the deepest level, which guarantees
       every path reaches the sink;
    3. random level-consistent arcs are added until the target complexity
       ``NC * (n + 2)`` is reached.
    """
    n = spec.n_activities
    nodes = n + 2
    sink = nodes - 1
    levels = spec.levels
    if levels < 2:
        raise ValueError("need at least two precedence levels")

    # Contiguous, non-empty level assignment for the real activities, then a
    # random relabelling so activity ids carry no level information.
    level_of = np.empty(nodes, dtype=np.int64)
    level_of[0] = 0
    level_of[sink] = levels + 1
    real = rng.permutation(n) + 1
    level_of[real] = 1 + (np.arange(n) * levels) // n

    by_level: list[list[int]] = [[] for _ in range(levels + 2)]
    for node in range(nodes):
        by_level[int(level_of[node])].append(node)

    arcs: set[tuple[int, int]] = set()
    has_successor = [False] * nodes
    for level in range(1, levels + 1):
        for node in by_level[level]:
            parent = 0 if level == 1 else int(rng.choice(by_level[level - 1]))
            arcs.add((parent, node))
            has_successor[parent] = True

    for level in range(levels, 0, -1):
        for node in by_level[level]:
            if has_successor[node]:
                continue
            child = sink if level == levels else int(rng.choice(by_level[level + 1]))
            arcs.add((node, child))
            has_successor[node] = True

    # Pad with random level-consistent arcs until the target complexity.
    target = max(len(arcs), int(round(spec.network_complexity * nodes)))
    attempts = 0
    while len(arcs) < target and attempts < 50 * target:
        attempts += 1
        low, high = sorted(int(v) for v in rng.choice(nodes, size=2, replace=False))
        if int(level_of[low]) >= int(level_of[high]):
            continue
        arcs.add((low, high))
    if len(arcs) < target:
        raise RuntimeError(
            f"could not reach network complexity {spec.network_complexity} "
            f"({len(arcs)}/{target} arcs) at depth {levels}"
        )

    lists: list[list[int]] = [[] for _ in range(nodes)]
    for source, target_node in sorted(arcs):
        lists[source].append(target_node)
    return lists


def generate(spec: GeneratorSpec, seed: int) -> RCPSPInstance:
    """Generate one single-project RCPSP instance.

    Node 0 is the dummy source and node ``n + 1`` the dummy sink, matching the
    convention of ``src.data.parsers``.  ``RS`` is realised exactly: the
    measured resource strength of the result equals ``spec.resource_strength``
    up to integer rounding.
    """
    rng = np.random.default_rng(seed)
    n = spec.n_activities
    resources = spec.n_resources
    nodes = n + 2

    successors = _build_precedence(spec, rng)

    durations = np.zeros(nodes, dtype=np.int64)
    durations[1 : n + 1] = rng.integers(1, spec.max_duration + 1, size=n)

    demands = np.zeros((nodes, resources), dtype=np.int64)
    used = rng.random((n, resources)) < spec.resource_factor
    demands[1 : n + 1] = np.where(
        used, rng.integers(1, spec.max_demand + 1, size=(n, resources)), 0
    )

    capacities = np.zeros(resources, dtype=np.int64)
    for r in range(resources):
        column = demands[1 : n + 1, r]
        r_min = int(column.max()) if n else 0
        r_max = int(column.sum())
        if r_max <= r_min:
            capacities[r] = max(r_min, 1)
        else:
            capacities[r] = max(
                r_min, int(round(r_min + spec.resource_strength * (r_max - r_min)))
            )

    return RCPSPInstance(
        name=spec.label(),
        n_activities=nodes,
        n_renewable=resources,
        horizon=int(durations.sum()),
        durations=durations,
        demands=demands,
        capacities=capacities,
        successors=successors,
    )


def write_rcp(instance: RCPSPInstance, path) -> None:
    """Serialise an instance in the OR&S / ProGen ``.rcp`` format.

    Layout (as parsed by ``src.data.parsers.parse_rcp``)::

        n_jobs n_renew
        capacity_1 .. capacity_R
        duration_j  demand_j1 .. demand_jR  n_successors  successor_ids (1-indexed)
    """
    lines = [f"{instance.n_activities} {instance.n_renewable}"]
    lines.append(" ".join(str(int(value)) for value in instance.capacities))
    lines.append("")
    for job in range(instance.n_activities):
        successors = [str(int(s) + 1) for s in instance.successors[job]]
        row = [str(int(instance.durations[job]))]
        row += [str(int(value)) for value in instance.demands[job]]
        row.append(str(len(successors)))
        row += successors
        lines.append(" ".join(row))
    text = "\n".join(lines) + "\n"
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
