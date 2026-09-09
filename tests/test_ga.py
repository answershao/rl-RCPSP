"""Random-key GA: determinism, no-worse-than-history, decode validity."""

import pytest

from src.core.ga import GAParameters, evaluate, run_ga
from src.data.adapter import load_core_instance

N_INSTANCES = 3
SEEDS = (20, 60, 100)  # distinct seeds avoid identical initial populations on j30


@pytest.fixture(scope="module")
def ga_instances(repo_root):
    files = sorted((repo_root / "data/psplib/j30").glob("*.sm"))[:N_INSTANCES]
    assert len(files) == N_INSTANCES
    return [(path, load_core_instance(path)) for path in files]


def test_ga_determinism_and_monotonicity(ga_instances):
    params = GAParameters(population=20, generations=30, tournament=3, elite=2)
    for (path, instance), seed in zip(ga_instances, SEEDS):
        first = run_ga(instance, seed=seed, parameters=params)
        second = run_ga(instance, seed=seed, parameters=params)
        assert first.best_makespan == second.best_makespan, f"nondeterministic GA on {path.name}"
        assert first.history == second.history, f"history differs on {path.name}"
        # Elitism: the tracked best can never worsen across generations.
        assert all(b <= a for a, b in zip(first.history, first.history[1:])), (
            f"best makespan worsened mid-run on {path.name}"
        )
        assert first.best_makespan == min(first.history)
        # Decoding the best keys reproduces the recorded makespan via the validated SGS.
        assert evaluate(instance, first.best_keys) == first.best_makespan
        assert first.evaluations == params.population * (params.generations + 1)
