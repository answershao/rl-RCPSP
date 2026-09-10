"""Tests for the ProGen-style instance generator (``src.data.generator``)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.adapter import to_core_instance
from src.data.generator import GeneratorSpec, depth_for, generate, write_rcp
from src.data.parsers import load_instance

REACHABILITY_SEEDS = range(12)


def _critical_path(instance) -> float:
    durations = np.asarray(instance.durations, dtype=float)
    earliest = np.zeros(instance.n_activities, dtype=float)
    for node in instance.topological_order():
        for successor in instance.successors[node]:
            earliest[successor] = max(
                earliest[successor], earliest[node] + durations[node]
            )
    return float(earliest.max())


def _reachable_from(instance, start: int) -> set[int]:
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()
        for successor in instance.successors[node]:
            if successor not in seen:
                seen.add(successor)
                stack.append(successor)
    return seen


@pytest.mark.parametrize(
    "spec",
    [
        GeneratorSpec(30, 0.75, 0.05, 1.6),
        GeneratorSpec(30, 0.25, 0.30, 2.1),
        GeneratorSpec(60, 1.0, 0.15, 1.8),
        GeneratorSpec(120, 0.5, 0.08, 1.5),
    ],
)
def test_realised_parameters_match_the_request(spec: GeneratorSpec) -> None:
    """RF, RS and NC must be hit by construction, not merely approached."""
    rfs, rss, arcs = [], [], []
    for seed in range(25):
        instance = generate(spec, seed)
        durations = np.asarray(instance.durations, dtype=float)
        demands = np.asarray(instance.demands, dtype=float)
        capacities = np.asarray(instance.capacities, dtype=float)
        real = durations > 0
        real_demands = demands[real]

        rfs.append(float((real_demands > 0).sum(axis=1).mean() / len(capacities)))
        terms = []
        for r in range(len(capacities)):
            column = real_demands[:, r]
            spread = column.sum() - column.max()
            terms.append(
                (capacities[r] - column.max()) / spread if spread > 0 else 1.0
            )
        rss.append(float(np.mean(terms)))
        arcs.append(
            (sum(len(s) for s in instance.successors), instance.n_activities)
        )

    # RF is a Bernoulli draw, so it is only statistically exact.
    assert abs(np.mean(rfs) - spec.resource_factor) < 0.03
    assert abs(np.mean(rss) - spec.resource_strength) < 0.01
    # NC is exact: the arc count lands on round(NC * nodes).
    for arc_count, nodes in arcs:
        assert arc_count == int(round(spec.network_complexity * nodes))


def test_network_complexity_below_the_skeleton_is_raised_not_ignored() -> None:
    """A connectivity skeleton needs ~1 arc per node; NC under that is clamped up."""
    spec = GeneratorSpec(120, 0.5, 0.08, 1.2)
    instance = generate(spec, 3)
    arcs = sum(len(s) for s in instance.successors)
    assert arcs >= int(round(spec.network_complexity * instance.n_activities))
    assert arcs / instance.n_activities > spec.network_complexity


def test_precedence_graph_is_connected_and_acyclic() -> None:
    """Every activity must be reachable from the source and reach the sink."""
    for seed in REACHABILITY_SEEDS:
        instance = generate(GeneratorSpec(45, 0.75, 0.1, 1.8), seed)
        sink = instance.n_activities - 1
        forward = _reachable_from(instance, 0)
        assert len(forward) == instance.n_activities

        predecessors: list[list[int]] = [[] for _ in range(instance.n_activities)]
        for node, successors in enumerate(instance.successors):
            for successor in successors:
                predecessors[successor].append(node)
        seen, stack = {sink}, [sink]
        while stack:
            node = stack.pop()
            for parent in predecessors[node]:
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        assert len(seen) == instance.n_activities


def test_every_activity_fits_within_capacity() -> None:
    for seed in range(10):
        instance = generate(GeneratorSpec(60, 1.0, 0.05, 2.1), seed)
        assert np.all(instance.demands <= instance.capacities[None, :])


def test_depth_tracks_psplib_critical_path() -> None:
    """Calibrated so the critical path is in PSPLIB's ballpark, not 2x it."""
    for n, expected in ((30, 41.0), (60, 72.0), (90, 87.0), (120, 111.0)):
        spec = GeneratorSpec(n, 0.75, 0.05, 1.8)
        measured = np.mean(
            [_critical_path(generate(spec, seed)) for seed in range(8)]
        )
        assert abs(measured - expected) / expected < 0.15, (n, measured, expected)
        assert depth_for(n) == spec.levels


def test_generation_is_deterministic() -> None:
    spec = GeneratorSpec(40, 0.5, 0.12, 1.8)
    first, second = generate(spec, 12345), generate(spec, 12345)
    assert np.array_equal(first.durations, second.durations)
    assert np.array_equal(first.demands, second.demands)
    assert np.array_equal(first.capacities, second.capacities)
    assert first.successors == second.successors

    other = generate(spec, 12346)
    assert first.successors != other.successors


def test_rcp_round_trip(tmp_path) -> None:
    """The written file must parse back to the same instance."""
    spec = GeneratorSpec(35, 0.75, 0.08, 1.9)
    original = generate(spec, 7)
    path = tmp_path / "sample.rcp"
    write_rcp(original, path)
    reloaded = load_instance(path)

    assert reloaded.n_activities == original.n_activities
    assert reloaded.n_renewable == original.n_renewable
    assert np.array_equal(reloaded.durations, original.durations)
    assert np.array_equal(reloaded.demands, original.demands)
    assert np.array_equal(reloaded.capacities, original.capacities)
    assert [sorted(s) for s in reloaded.successors] == [
        sorted(s) for s in original.successors
    ]
    # and it must survive the adapter the RL stack uses
    core = to_core_instance(reloaded)
    assert len(core.activities) == original.n_activities


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_activities": 1, "resource_factor": 0.5, "resource_strength": 0.5, "network_complexity": 1.5},
        {"n_activities": 30, "resource_factor": 0.0, "resource_strength": 0.5, "network_complexity": 1.5},
        {"n_activities": 30, "resource_factor": 1.5, "resource_strength": 0.5, "network_complexity": 1.5},
        {"n_activities": 30, "resource_factor": 0.5, "resource_strength": 1.5, "network_complexity": 1.5},
        {"n_activities": 30, "resource_factor": 0.5, "resource_strength": 0.5, "network_complexity": 0.0},
        {"n_activities": 30, "resource_factor": 0.5, "resource_strength": 0.5, "network_complexity": 1.5, "n_resources": 0},
    ],
)
def test_invalid_specs_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        GeneratorSpec(**kwargs)
