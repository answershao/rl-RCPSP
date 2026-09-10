"""Generate a candidate PPO training pool that covers the PSPLIB parameter grid.

The problem this solves
-----------------------
``splits.json`` currently makes RG30 the only training pool.  RG30 is a *point*
in generator-parameter space (RF fixed at 0.75, RS in [0.003, 0.046], n fixed at
30), while PSPLIB j30-j120 is a designed factorial experiment over (RF, RS, NC)
at four sizes: 48 cells for j30/j60/j90 (4 RF x 4 RS x 3 NC) and 60 for j120
(4 x 5 x 3).  Training on a point and testing on a grid is exactly the
distribution shift measured by ``scripts/instance_stats.py``.

This script closes that gap *without touching the test instances*:

1. read the realised ``(n, RF, RS, NC)`` coordinate of every PSPLIB set
   (one coordinate per ``.bas`` file, averaged over its 10 instances);
2. recover the discrete factor levels -- RF and NC are already discrete, RS is
   clustered into ``cells / (|RF| * |NC|)`` equal-frequency levels, which
   recovers the generator's RS axis;
3. rebuild the full factorial grid per size and generate fresh instances at
   every cell with independent seeds;
4. optionally extend the grid one step beyond the observed range on each axis,
   so the benchmark cells become *interior* points rather than sitting on the
   boundary of the training envelope;
5. emit a manifest in the ``read_protocol`` schema, so the pool can be plugged
   into ``scripts/train_ppo.py --splits`` unchanged.

Two modes:

``--mode psp-grid``
    Grid recovered from PSPLIB -- same distribution as the test set, disjoint
    instances.  The defensible way to say "trained on the test distribution
    without touching the test set".

``--mode random``
    Coordinates sampled uniformly from explicit ranges (domain randomization).
    Unbounded data, and the natural fit for an on-the-fly sampler.

Usage:
    python -m scripts.generate_pool --mode psp-grid --replicates 5 --widen \
        --workers 8 --output data/generated/psp_grid \
        --manifest data/generated/psp_grid.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.common import find_instances, map_jobs
from scripts.instance_stats import instance_parameters
from src.data.adapter import load_core_instance
from src.data.generator import GeneratorSpec, generate, write_rcp

# Suite id -> activity count, for the four PSPLIB sizes.
PSPLIB_SIZES = {"psplib_j30": 30, "psplib_j60": 60, "psplib_j90": 90, "psplib_j120": 120}


def coordinate_row(path: Path, suite: str) -> dict:
    """``(n, RF, RS, NC)`` for one instance; picklable for the spawn pool."""
    params = instance_parameters(load_core_instance(path))
    return {
        "suite": suite,
        "set_id": path.stem.split("_")[0],
        "n": params["n"],
        "RF": params["RF"],
        "RS": params["RS"],
        "NC": params["NC"],
    }


def psp_cell_coordinates(
    data_root: Path, workers: int
) -> list[tuple[int, float, float, float]]:
    """One averaged ``(n, RF, RS, NC)`` coordinate per PSPLIB set."""
    jobs = [
        (path, suite) for suite in PSPLIB_SIZES for path in find_instances(data_root, suite)
    ]
    rows = map_jobs(coordinate_row, jobs, workers)

    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_cell.setdefault((row["suite"], row["set_id"]), []).append(row)

    coordinates = []
    for (suite, _set_id), members in by_cell.items():
        coordinates.append(
            (
                int(PSPLIB_SIZES[suite]),
                float(np.mean([m["RF"] for m in members])),
                float(np.mean([m["RS"] for m in members])),
                float(np.mean([m["NC"] for m in members])),
            )
        )
    return sorted(coordinates)


def _cluster_levels(values: np.ndarray, k: int) -> list[float]:
    """Split sorted 1-D values into ``k`` equal-frequency levels, return means."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if k <= 1:
        return [float(ordered.mean())]
    return [float(group.mean()) for group in np.array_split(ordered, k)]


def extract_levels(
    coordinates: list[tuple[int, float, float, float]]
) -> dict[int, dict[str, list[float]]]:
    """Recover the discrete RF / RS / NC levels PSPLIB was generated from."""
    levels: dict[int, dict[str, list[float]]] = {}
    for size in sorted({c[0] for c in coordinates}):
        cells = [c for c in coordinates if c[0] == size]
        rf = sorted({round(c[1], 3) for c in cells})
        nc = sorted({round(c[3], 3) for c in cells})
        rs_count = max(1, len(cells) // (len(rf) * len(nc)))
        rs = _cluster_levels([c[2] for c in cells], rs_count)
        levels[size] = {"rf": rf, "rs": rs, "nc": nc}
    return levels


def build_grid(
    levels: dict[int, dict[str, list[float]]],
    *,
    rs_extra: tuple[float, ...] = (),
    nc_extra: tuple[float, ...] = (),
) -> list[tuple[int, float, float, float]]:
    """Full factorial grid, optionally extended past the observed range."""
    grid: list[tuple[int, float, float, float]] = []
    for size, axes in levels.items():
        for rf in axes["rf"]:
            for rs in list(axes["rs"]) + list(rs_extra):
                for nc in list(axes["nc"]) + list(nc_extra):
                    grid.append((size, float(rf), float(rs), float(nc)))
    return grid


def random_coordinates(
    rng: np.random.Generator, sizes: list[int], count: int
) -> list[tuple[int, float, float, float]]:
    """Sample coordinates uniformly from explicit ranges."""
    return [
        (
            int(rng.choice(sizes)),
            float(rng.choice([0.25, 0.5, 0.75, 1.0])),
            float(rng.uniform(0.01, 0.35)),
            float(rng.choice([1.5, 1.8, 2.1])),
        )
        for _ in range(count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("psp-grid", "random"), default="psp-grid")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/psp_grid"),
        help="directory the generated .rcp files are written to",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/psp_grid.json"),
        help="manifest in the read_protocol schema, loadable via --splits",
    )
    parser.add_argument("--replicates", type=int, default=5, help="instances per grid cell")
    parser.add_argument(
        "--validation-replicates",
        type=int,
        default=1,
        help="instances per grid cell held out for training-period validation",
    )
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("splits.json"),
        help="protocol whose evaluation groups the manifest should carry over, so "
        "the generated pool is a complete, pluggable protocol",
    )
    parser.add_argument(
        "--random-count", type=int, default=2000, help="cells to sample in --mode random"
    )
    parser.add_argument(
        "--widen",
        action="store_true",
        help="add one factor level beyond the observed range on the RS and NC axes",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.mode == "psp-grid":
        cells = psp_cell_coordinates(args.data_root, args.workers)
        levels = extract_levels(cells)
        for size, axes in levels.items():
            print(
                f"n={size}: RF {[round(v, 3) for v in axes['rf']]} "
                f"RS {[round(v, 4) for v in axes['rs']]} "
                f"NC {[round(v, 3) for v in axes['nc']]}"
            )
        rs_extra: tuple[float, ...] = ()
        nc_extra: tuple[float, ...] = ()
        if args.widen:
            rs_extra = (round(max(max(a["rs"]) for a in levels.values()) * 1.6, 3),)
            nc_extra = (round(max(max(a["nc"]) for a in levels.values()) + 0.3, 3),)
        coordinates = build_grid(levels, rs_extra=rs_extra, nc_extra=nc_extra)
        print(f"grid cells: {len(coordinates)} (observed PSPLIB sets: {len(cells)})")
    else:
        coordinates = random_coordinates(
            rng, sorted(PSPLIB_SIZES.values()), args.random_count
        )

    train_entries: list[str] = []
    validation_entries: list[str] = []
    specs: list[dict] = []
    root = args.output.relative_to(args.data_root)

    for cell_index, (n, rf, rs, nc) in enumerate(coordinates):
        spec = GeneratorSpec(n, rf, rs, nc)
        for replicate in range(args.replicates):
            seed = int(rng.integers(0, 2**31 - 1))
            instance = generate(spec, seed)
            name = f"n{n}/c{cell_index:04d}_r{replicate:02d}"
            write_rcp(instance, args.data_root / root / f"{name}.rcp")
            rel = (root / f"{name}.rcp").as_posix()
            is_validation = replicate < args.validation_replicates
            (validation_entries if is_validation else train_entries).append(rel)
            specs.append(
                {
                    "file": rel,
                    "split": "validation" if is_validation else "train",
                    "requested": {"n": n, "RF": rf, "RS": rs, "NC": nc},
                    "seed": seed,
                }
            )

    manifest = {
        "protocol": (
            "generated training pool; 'rg30_train' / 'rg30_validation' are the "
            "split keys expected by src.data.instances.read_protocol, not a "
            "statement about the RG30 corpus"
        ),
        "generator": "src/data/generator.py",
        "mode": args.mode,
        "seed": args.seed,
        "replicates": args.replicates,
        "widen": bool(args.widen),
        "splits": {
            "rg30_train": sorted(train_entries),
            "rg30_validation": sorted(validation_entries),
        },
        "evaluation": json.loads(args.splits.read_text()).get("evaluation", {}),
        "counts": {
            "train": len(train_entries),
            "validation": len(validation_entries),
            "cells": len(coordinates),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1))
    args.manifest.with_suffix(".specs.json").write_text(json.dumps(specs, indent=1))

    print(f"mode={args.mode} cells={len(coordinates)} replicates={args.replicates}")
    print(f"train={len(train_entries)} validation={len(validation_entries)}")
    print(f"wrote {args.output} and {args.manifest}")


if __name__ == "__main__":
    main()
