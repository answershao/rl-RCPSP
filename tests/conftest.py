"""Shared fixtures for the RCPSP test-suite (repo / data-sample paths)."""

from pathlib import Path

import pytest

from tests import REPO_ROOT, TEST_INSTANCE, TEST_INSTANCE_2


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root (parent of ``src/``, ``data/``, ``splits.json``)."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def j30_sample() -> Path:
    """Real single-project j30 instance shared by env / RL tests."""
    return TEST_INSTANCE


@pytest.fixture(scope="session")
def j30_sample_2() -> Path:
    return TEST_INSTANCE_2
