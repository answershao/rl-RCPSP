"""Guards for the suite matrix in ``scripts.common``.

The training pool must stay registered as a suite.  ``scripts/train_ppo`` picks
its checkpoint on the ``validation`` split using reference-rule makespans from
``scripts/baselines``, and that runner can only discover instances through
``SUITE_SPECS``.  Deregistering the pool -- or pointing its directory somewhere
else -- breaks training with a confusing "does not cover N validation
instances" error, so the invariant is asserted here instead of being
rediscovered.  See docs/EXECUTION_FLOW.md S3.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.common import (
    DEFAULT_SUITES,
    EVALUATION_SUITES,
    SUITE_SPECS,
    resolve_suite_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_POOL = "psp_grid"


def test_training_pool_is_registered_as_a_suite() -> None:
    directory, pattern, role = SUITE_SPECS[TRAINING_POOL]
    assert (directory, pattern, role) == (
        "generated/psp_grid_bal",
        "**/*.rcp",
        "training-pool",
    )


def test_default_suites_is_the_training_pool() -> None:
    # S3 needs no extra flags: `python -m scripts.baselines` alone covers the pool.
    assert DEFAULT_SUITES == TRAINING_POOL
    assert resolve_suite_ids(DEFAULT_SUITES) == [TRAINING_POOL]


def test_evaluation_suites_exclude_every_training_pool() -> None:
    # A rule/GA/GPHH default that included a training pool would report scores on
    # the very instances the method was fitted to.
    roles = {SUITE_SPECS[suite][2] for suite in EVALUATION_SUITES}
    assert "training-pool" not in roles
    assert TRAINING_POOL not in EVALUATION_SUITES
    assert EVALUATION_SUITES == (
        "psplib_j30",
        "psplib_j60",
        "psplib_j90",
        "psplib_j120",
    )


def test_training_pool_directory_covers_the_validation_split() -> None:
    """The pool must be able to yield the reference CSV train_ppo demands."""
    manifest = json.loads((REPO_ROOT / "splits.json").read_text())
    directory, _, _ = SUITE_SPECS[TRAINING_POOL]
    prefix = f"{directory}/"
    validation = manifest["splits"]["validation"]
    assert validation, "the protocol has an empty validation split"
    outside = [rel for rel in validation if not rel.startswith(prefix)]
    assert not outside, (
        f"{len(outside)}/{len(validation)} validation instances live outside "
        f"{prefix}; train_ppo's --ref-rules could never cover them, e.g. {outside[:2]}"
    )


def test_generated_validation_split_is_parameter_stratified() -> None:
    manifest = json.loads((REPO_ROOT / "splits.json").read_text())
    assert manifest["validation_per_size"] == 32
    assert manifest["validation_strategy"] == "stratified_rf_rs_nc_per_size"
    specs = json.loads(
        (REPO_ROOT / "data/generated/psp_grid_bal.specs.json").read_text()
    )
    paths_by_split = {
        split: {entry["file"] for entry in specs if entry["split"] == split}
        for split in ("train", "validation")
    }
    assert paths_by_split == {
        split: set(manifest["splits"][split]) for split in paths_by_split
    }
    assert {split: len(paths) for split, paths in paths_by_split.items()} == {
        "train": 1088,
        "validation": 128,
    }

    for size in (30, 60, 90, 120):
        all_at_size = [entry for entry in specs if entry["nominal"]["n"] == size]
        held_out = [entry for entry in all_at_size if entry["split"] == "validation"]
        assert len(held_out) == 32
        for axis in ("RF", "RS", "NC"):
            all_levels = {entry["nominal"][axis] for entry in all_at_size}
            counts = Counter(entry["nominal"][axis] for entry in held_out)
            assert set(counts) == all_levels
            assert max(counts.values()) - min(counts.values()) <= 1
