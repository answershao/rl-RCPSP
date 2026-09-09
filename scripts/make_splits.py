"""Generate the reproducible data split manifest for the 4-suite RCPSP protocol.

Roles (user-confirmed protocol):
  - RG30           : training pool only. ~10% per Set directory is held out as a
                     *training-period* validation split (checkpoint / seed selection).
  - PSPLIB j30-120 : final-evaluation benchmark only (gap vs BKS). Never used in
                     training or training-period validation.
  - RG300 / Patterson : final-evaluation only (generalization / supplementary).

The manifest stores relative POSIX paths (project-root based) so every consumer
(baselines, PPO training, GA, GPHH) reads exactly the same file lists. Sampling
is stratified by the immediate parent directory of each RG30 instance and is
deterministic for a fixed ``--seed``.

Usage:
    python -m scripts.make_splits --data-root data --seed 20260909 \
        --val-fraction 0.1 --output splits.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from scripts.common import SUITE_SPECS, find_instances, natural_key, relative_posix


def make_rg30_splits(
    root: Path, *, val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Hold out ceil(fraction) instances per Set directory of RG30.

    Stratifying by the immediate parent (Set 1 .. Set 5) keeps the validation
    split representative of the whole training pool instead of one set.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    files = find_instances(root, "rg30")
    by_stratum: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_stratum[path.parent.name].append(path)

    rng = random.Random(seed)
    train: list[str] = []
    validation: list[str] = []
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum], key=natural_key)
        holdout = max(1, int(__import__("math").ceil(val_fraction * len(members))))
        rng.shuffle(members)
        selected, remaining = members[:holdout], members[holdout:]
        validation.extend(relative_posix(root, p) for p in selected)
        train.extend(relative_posix(root, p) for p in sorted(remaining, key=natural_key))
    train.sort()
    validation.sort()
    return train, validation


def collect_evaluation_suite(root: Path, suite: str) -> list[str]:
    return [relative_posix(root, p) for p in find_instances(root, suite)]


def build_manifest(
    root: Path, *, val_fraction: float, seed: int
) -> dict:
    train, validation = make_rg30_splits(root, val_fraction=val_fraction, seed=seed)
    evaluation = {
        suite: collect_evaluation_suite(root, suite)
        for suite, (_, _, role) in SUITE_SPECS.items()
        if role == "final-evaluation"
    }
    counts = {
        "rg30_train": len(train),
        "rg30_validation": len(validation),
        **{f"eval_{suite}": len(paths) for suite, paths in evaluation.items()},
    }
    total = sum(counts.values())
    return {
        "protocol": (
            "RG30 is the only training pool; PSPLIB j30-j120, RG300 and Patterson are "
            "final-evaluation only and must never appear in training or in the "
            "training-period validation split."
        ),
        "generator": "scripts/make_splits.py",
        "seed": seed,
        "val_fraction_per_set": val_fraction,
        "splits": {
            "rg30_train": train,
            "rg30_validation": validation,
        },
        "evaluation": evaluation,
        "counts": counts,
        "total_instances": total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("splits.json"))
    return parser.parse_args()


def verify(manifest: dict, root: Path) -> None:
    expected = {
        "rg30_train": 1800 - 180,
        "rg30_validation": 180,
        "eval_psplib_j30": 480,
        "eval_psplib_j60": 480,
        "eval_psplib_j90": 480,
        "eval_psplib_j120": 600,
        "eval_rg300": 480,
        "eval_patterson": 110,
    }
    for key, want in expected.items():
        got = manifest["counts"][key]
        if got != want:
            raise AssertionError(f"{key}: expected {want}, got {got}")
    train = set(manifest["splits"]["rg30_train"])
    validation = set(manifest["splits"]["rg30_validation"])
    if train & validation:
        raise AssertionError("rg30_train and rg30_validation must be disjoint")
    seen = set()
    for suite, paths in manifest["evaluation"].items():
        for p in paths:
            if p in seen or p in train or p in validation:
                raise AssertionError(f"duplicated instance across splits: {p}")
            seen.add(p)


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.data_root, val_fraction=args.val_fraction, seed=args.seed)
    verify(manifest, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"total: {manifest['total_instances']}")


if __name__ == "__main__":
    main()
