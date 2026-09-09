"""Priority-rule baselines: determinism across the serial/parallel column set."""

import pytest

from src.core.rules import all_makespans, available_columns, rule_makespan
from src.data.adapter import load_core_instance

N_INSTANCES = 5
SEED = 17


@pytest.fixture(scope="module")
def rule_instances(repo_root):
    """First N j30 instances as core-format instances (path, instance) pairs."""
    files = sorted((repo_root / "data/psplib/j30").glob("*.sm"))[:N_INSTANCES]
    assert len(files) == N_INSTANCES
    return [(path, load_core_instance(path)) for path in files]


def test_column_contract():
    columns = available_columns()
    assert "serial_LST" in columns
    assert "parallel_WCS" in columns
    assert "serial_WCS" not in columns  # WCS is a parallel-SGS-only rule


def test_determinism_across_columns(rule_instances):
    columns = available_columns()
    for path, instance in rule_instances:
        first = all_makespans(instance, seed=SEED)
        second = all_makespans(instance, seed=SEED)
        assert first == second, f"deterministic rerun mismatch for {path.name}"
        assert set(first) == set(columns), f"column set mismatch for {path.name}"


def test_wcs_serial_rejected(rule_instances):
    _, instance = rule_instances[0]
    with pytest.raises(ValueError, match="parallel SGS only"):
        rule_makespan(instance, "WCS", scheme="serial")


def test_unknown_rule_and_scheme_raise_with_guidance(rule_instances):
    _, instance = rule_instances[0]
    # Invalid rule name: error must name the offending rule and the valid set.
    with pytest.raises(ValueError, match=r"unknown static rule: 'NOPE'.*choose from"):
        rule_makespan(instance, "NOPE")
    # Invalid scheme: error must name the valid schemes.
    with pytest.raises(ValueError, match=r"unknown scheme: 'bogus'.*serial.*parallel"):
        rule_makespan(instance, "FIFO", scheme="bogus")
