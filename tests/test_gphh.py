"""GP hyper-heuristic: determinism, monotone history, valid terminals/rules."""

import random as _random

import pytest

from src.core.gphh import (
    GPHHParameters,
    depth,
    evaluate_tree,
    random_tree,
    rule_makespan,
    run_gphh,
)
from src.core.rules import activity_features
from src.data.adapter import load_core_instance

N_INSTANCES = 3


def _collect_terminals(node, out):
    if isinstance(node, str):
        out.add(node)
    elif not isinstance(node, (int, float)):
        for child in node[1:]:
            _collect_terminals(child, out)


@pytest.fixture(scope="module")
def gphh_instances(repo_root):
    files = sorted((repo_root / "data/psplib/j30").glob("*.sm"))[:N_INSTANCES]
    assert len(files) == N_INSTANCES
    return [load_core_instance(path) for path in files]


def test_gphh_reproducible_and_valid(gphh_instances):
    params = GPHHParameters(population=20, generations=12, tournament=3, elite=2, max_depth=5)
    first = run_gphh(gphh_instances, seed=7, parameters=params)
    second = run_gphh(gphh_instances, seed=7, parameters=params)
    assert first.best_rule == second.best_rule, "GPHH must be seed-reproducible"
    assert first.history == second.history, "history must be seed-reproducible"
    assert all(b <= a for a, b in zip(first.history, first.history[1:])), (
        "elitism must never worsen the best training fitness"
    )
    assert first.best_fitness == min(first.history)
    assert depth(first.best_rule) <= params.max_depth

    # The best rule must decode to a valid schedule on every training instance.
    for instance in gphh_instances:
        assert rule_makespan(instance, first.best_rule) > 0, (
            f"invalid best-rule makespan on {instance.name}"
        )


def test_random_tree_terminals_and_scores(gphh_instances):
    sample = activity_features(gphh_instances[0])[next(iter(gphh_instances[0].activities))]
    rng = _random.Random(3)
    for _ in range(50):
        tree = random_tree(rng, 4)
        used = set()
        _collect_terminals(tree, used)
        for name in used:
            assert name in sample, f"unknown terminal {name!r} in {tree}"
        assert isinstance(evaluate_tree(tree, sample), float)
