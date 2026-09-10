"""Generate a candidate PPO training pool that covers the PSPLIB parameter grid.

The problem this solves
-----------------------
PSPLIB j30-j120 is a designed factorial experiment over (RF, RS, NC)
at four sizes: 48 cells for j30/j60/j90 (4 RF x 4 RS x 3 NC) and 60 for j120
(4 x 5 x 3).  Training on a point and testing on a grid is exactly the
distribution shift measured by ``scripts/instance_stats.py``.

This script closes that gap *without touching the test instances*:

1. read the nominal factor design (NC / RF / RS levels per set) from the
   RCPLIB workbook (``scripts/psplib_design.py``) -- authoritative, no
   clustering heuristics;
2. measure every PSPLIB set's ``(RF, RS, NC)`` *in this repo's own parameter
   definitions* (the definitions ``src/data/generator.py`` targets and hits
   exactly; the workbook's RS uses Kolisch's original formula and must not be
   fed to the generator directly);
3. map each nominal level to its generator-space coordinate (mean measured
   value of the sets carrying that nominal level) and rebuild the full
   factorial grid;
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
    python -m scripts.generate_pool --mode psp-grid --replicates 8,4,2,1 --widen \
        --workers 8 --output data/generated/psp_grid \
        --manifest data/generated/psp_grid.json

``--replicates`` takes a single int (uniform quota) or one int per size
(n=30/60/90/120, ascending).  ``8,4,2,1`` balances each size's decision-state
budget, since the per-instance training cost grows roughly with n^2.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.common import EVALUATION_SUITES, find_instances, map_jobs
from scripts.instance_stats import instance_parameters
from scripts.psplib_design import load_psplib_design
from src.data.adapter import load_core_instance
from src.data.generator import GeneratorSpec, generate, write_rcp

# Suite id -> activity count, for the four PSPLIB sizes.
PSPLIB_SIZES = {"psplib_j30": 30, "psplib_j60": 60, "psplib_j90": 90, "psplib_j120": 120}

_SET_RE = re.compile(r"j(30|60|90|120)(\d+)_")


def coordinate_row(path: Path, suite: str) -> dict:
    """``(set_id, n, RF, RS, NC)`` for one instance; picklable for spawn pools."""
    params = instance_parameters(load_core_instance(path))
    match = _SET_RE.match(path.stem)
    return {
        "suite": suite,
        "set_id": int(match.group(2)) if match else -1,
        "n": params["n"],
        "RF": params["RF"],
        "RS": params["RS"],
        "NC": params["NC"],
    }


def psp_cell_coordinates(
    data_root: Path, workers: int
) -> dict[tuple[int, int], tuple[float, float, float]]:
    """One averaged ``(RF, RS, NC)`` coordinate per PSPLIB set, in this repo's
    parameter definitions (what ``GeneratorSpec`` targets)."""
    jobs = [
        (path, suite) for suite in PSPLIB_SIZES for path in find_instances(data_root, suite)
    ]
    rows = map_jobs(coordinate_row, jobs, workers)

    by_cell: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        by_cell.setdefault((row["suite"], row["set_id"]), []).append(row)

    coordinates: dict[tuple[int, int], tuple[float, float, float]] = {}
    for (suite, set_id), members in by_cell.items():
        coordinates[(PSPLIB_SIZES[suite], set_id)] = (
            float(np.mean([m["RF"] for m in members])),
            float(np.mean([m["RS"] for m in members])),
            float(np.mean([m["NC"] for m in members])),
        )
    return coordinates


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


def levels_from_design(
    design: list,
    cells_by_set: dict[tuple[int, int], tuple[float, float, float]],
) -> dict[int, dict[str, dict[float, float]]]:
    """Map every nominal design level to its generator-space coordinate.

    Returns, per size, ``{"rf": {nominal: measured}, "rs": ..., "nc": ...}``
    where "measured" is the mean of the set-level coordinates carrying that
    nominal level.  The generator cannot hit Kolisch's RS definition directly,
    so the nominal axis only supplies the *structure*; the coordinates stay in
    this repo's definitions.
    """
    levels: dict[int, dict[str, dict[float, float]]] = {}
    for size in sorted({s.size for s in design}):
        members = [s for s in design if s.size == size and s.key in cells_by_set]
        grouped: dict[str, dict[float, list[float]]] = {"rf": {}, "rs": {}, "nc": {}}
        for s in members:
            meas = cells_by_set[s.key]
            grouped["rf"].setdefault(s.rf, []).append(meas[0])
            grouped["rs"].setdefault(s.rs, []).append(meas[1])
            grouped["nc"].setdefault(s.nc, []).append(meas[2])
        levels[size] = {
            axis: {nom: float(np.mean(vals)) for nom, vals in sorted(by_nom.items())}
            for axis, by_nom in grouped.items()
        }
    return levels


def levels_from_clustering(
    cells_by_set: dict[tuple[int, int], tuple[float, float, float]],
) -> dict[int, dict[str, dict[float, float]]]:
    """Fallback when the RCPLIB workbook is unavailable: recover the RS axis by
    equal-frequency clustering, with nominal labels equal to the measured
    values (so the manifest simply carries the generator-space coordinates)."""
    coordinates = [
        (size, rf, rs, nc)
        for (size, _set_id), (rf, rs, nc) in sorted(cells_by_set.items())
    ]
    recovered = extract_levels(coordinates)
    return {
        size: {axis: {v: v for v in vals} for axis, vals in axes.items()}
        for size, axes in recovered.items()
    }


def build_grid(
    levels: dict[int, dict[str, dict[float, float]]],
    *,
    rs_extra: dict[float, float] | None = None,
    nc_extra: dict[float, float] | None = None,
) -> tuple[list[tuple[int, float, float, float]], list[dict]]:
    """Full factorial grid over the nominal levels, optionally extended past
    the observed range.

    Returns ``(coordinates, labels)``: generator-space ``(n, RF, RS, NC)``
    tuples plus the nominal design coordinate each cell was built from.
    """
    rs_extra = rs_extra or {}
    nc_extra = nc_extra or {}
    grid: list[tuple[int, float, float, float]] = []
    labels: list[dict] = []
    for size, axes in sorted(levels.items()):
        for rf_nom, rf in axes["rf"].items():
            for rs_nom, rs in list(axes["rs"].items()) + list(rs_extra.items()):
                for nc_nom, nc in list(axes["nc"].items()) + list(nc_extra.items()):
                    grid.append((size, float(rf), float(rs), float(nc)))
                    labels.append(
                        {"n": size, "RF": rf_nom, "RS": rs_nom, "NC": nc_nom}
                    )
    return grid, labels


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


def parse_replicates(value: str) -> dict[int, int]:
    """``--replicates`` accepts either ``5`` or a per-size list ``8,4,2,1``.

    A list maps to the PSPLIB sizes in ascending order (30/60/90/120); equal
    values mean the classic uniform quota, descending values balance each
    size's decision-state count (cost per instance grows ~n^2).
    """
    parts = [p.strip() for p in value.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(
            f"--replicates must be a positive int or 1-4 comma-separated positive ints, got {value!r}"
        )
    if len(values) == 1:
        return {size: values[0] for size in sorted(PSPLIB_SIZES.values())}
    if len(values) != len(PSPLIB_SIZES):
        raise argparse.ArgumentTypeError(
            f"--replicates list needs exactly {len(PSPLIB_SIZES)} values for sizes "
            f"{sorted(PSPLIB_SIZES.values())}, got {len(values)}"
        )
    return dict(zip(sorted(PSPLIB_SIZES.values()), values))


def stratified_validation_cells(
    labels: list[dict], fraction: float, seed: int, per_size: int = 0
) -> set[int]:
    """Select per-size holdout cells with balanced RF/RS/NC marginals."""
    selected: set[int] = set()
    rng = np.random.default_rng(seed)

    for size in sorted({label["n"] for label in labels}):
        candidates = [
            index for index, label in enumerate(labels) if label["n"] == size
        ]
        quota = per_size or max(1, math.ceil(len(candidates) * fraction))
        if quota > len(candidates):
            raise ValueError(
                f"validation quota {quota} exceeds {len(candidates)} cells for n={size}"
            )
        axes = ("RF", "RS", "NC")
        level_totals = {
            axis: Counter(labels[index][axis] for index in candidates)
            for axis in axes
        }
        targets = {
            axis: {
                level: quota * count / len(candidates)
                for level, count in totals.items()
            }
            for axis, totals in level_totals.items()
        }
        counts = {axis: Counter() for axis in axes}
        tie_break = {index: float(rng.random()) for index in candidates}
        remaining = set(candidates)

        for _ in range(quota):
            def score(index: int) -> tuple[float, float, int]:
                label = labels[index]
                squared_error = sum(
                    (counts[axis][level] + (label[axis] == level) - target) ** 2
                    for axis in axes
                    for level, target in targets[axis].items()
                )
                return squared_error, tie_break[index], index

            chosen = min(remaining, key=score)
            remaining.remove(chosen)
            selected.add(chosen)
            for axis in axes:
                counts[axis][labels[chosen][axis]] += 1

    return selected


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
    parser.add_argument(
        "--replicates",
        type=parse_replicates,
        default=parse_replicates("5"),
        metavar="N | N,N,N,N",
        help="instances per grid cell: a single int, or per-size counts for "
        "n=30/60/90/120 in ascending order (e.g. 8,4,2,1 balances the "
        "decision-state budget because per-instance cost grows ~n^2)",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="fraction of cells per size whose last replicate is held out for "
        "training-period validation; ignored when --validation-per-size is set",
    )
    parser.add_argument(
        "--validation-per-size",
        type=int,
        default=0,
        help="fixed number of cells held out for each activity size; 0 uses "
        "--validation-fraction",
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
        "--bks-xlsx",
        type=Path,
        default=Path("data/bks/RCPLIB (Parameters and BKS).xlsx"),
        help="RCPLIB workbook whose 'All' sheet carries the nominal PSPLIB "
        "factor design; if missing, the grid is recovered by clustering",
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
    labels: list[dict] | None = None
    if args.mode == "psp-grid":
        cells = psp_cell_coordinates(args.data_root, args.workers)
        if args.bks_xlsx.exists():
            design = load_psplib_design(args.bks_xlsx)
            levels = levels_from_design(design, cells)
            source = f"nominal design from {args.bks_xlsx}"
        else:
            levels = levels_from_clustering(cells)
            source = "equal-frequency clustering (workbook not found)"
        for size, axes in sorted(levels.items()):
            axes_text = " | ".join(
                f"{axis} nom->{ {k: round(v, 3) for k, v in mapping.items()} }"
                for axis, mapping in (
                    ("RF", axes["rf"]),
                    ("RS", axes["rs"]),
                    ("NC", axes["nc"]),
                )
            )
            print(f"n={size}: {axes_text}")
        print(f"levels recovered from {source}")
        rs_extra: dict[float, float] = {}
        nc_extra: dict[float, float] = {}
        if args.widen:
            rs_extra = {
                round(max(max(a["rs"]) for a in levels.values()) * 1.6, 3): round(
                    max(max(a["rs"].values()) for a in levels.values()) * 1.6, 3
                )
            }
            nc_extra = {
                round(max(max(a["nc"]) for a in levels.values()) + 0.3, 3): round(
                    max(max(a["nc"].values()) for a in levels.values()) + 0.3, 3
                )
            }
        coordinates, labels = build_grid(levels, rs_extra=rs_extra, nc_extra=nc_extra)
        print(f"grid cells: {len(coordinates)} (observed PSPLIB sets: {len(cells)})")
    else:
        coordinates = random_coordinates(
            rng, sorted(PSPLIB_SIZES.values()), args.random_count
        )

    train_entries: list[str] = []
    validation_entries: list[str] = []
    specs: list[dict] = []
    root = args.output.relative_to(args.data_root)

    if not 0.0 < args.validation_fraction <= 1.0:
        parser.error("--validation-fraction must be in (0, 1]")
    if args.validation_per_size < 0:
        parser.error("--validation-per-size must be non-negative")
    if labels is not None:
        try:
            validation_cells = stratified_validation_cells(
                labels,
                args.validation_fraction,
                args.seed,
                per_size=args.validation_per_size,
            )
        except ValueError as exc:
            parser.error(str(exc))
        validation_strategy = "stratified_rf_rs_nc_per_size"
    else:
        validation_cells = set()
        if args.validation_per_size:
            selection_rng = np.random.default_rng(args.seed)
            for size in sorted({coordinate[0] for coordinate in coordinates}):
                candidates = [
                    index
                    for index, coordinate in enumerate(coordinates)
                    if coordinate[0] == size
                ]
                if args.validation_per_size > len(candidates):
                    parser.error(
                        f"--validation-per-size exceeds cell count for n={size}"
                    )
                validation_cells.update(
                    selection_rng.choice(
                        candidates, size=args.validation_per_size, replace=False
                    ).tolist()
                )
        else:
            stride = max(1, round(1.0 / args.validation_fraction))
            seen_by_size: Counter[int] = Counter()
            for index, coordinate in enumerate(coordinates):
                size = coordinate[0]
                if seen_by_size[size] % stride == 0:
                    validation_cells.add(index)
                seen_by_size[size] += 1
        validation_strategy = "periodic_by_size"

    for cell_index, (coordinate, nominal) in enumerate(
        zip(coordinates, labels if labels is not None else [None] * len(coordinates))
    ):
        n, rf, rs, nc = coordinate
        spec = GeneratorSpec(n, rf, rs, nc)
        is_holdout_cell = cell_index in validation_cells
        replicates = args.replicates[n]
        for replicate in range(replicates):
            seed = int(rng.integers(0, 2**31 - 1))
            instance = generate(spec, seed)
            name = f"n{n}/c{cell_index:04d}_r{replicate:02d}"
            write_rcp(instance, args.data_root / root / f"{name}.rcp")
            rel = (root / f"{name}.rcp").as_posix()
            is_validation = is_holdout_cell and replicate == replicates - 1
            (validation_entries if is_validation else train_entries).append(rel)
            spec_entry = {
                "file": rel,
                "split": "validation" if is_validation else "train",
                "requested": {"n": n, "RF": rf, "RS": rs, "NC": nc},
                "seed": seed,
            }
            if nominal is not None:
                spec_entry["nominal"] = nominal
            specs.append(spec_entry)

    manifest = {
        "protocol": (
            "generated training pool; split keys 'train' / 'validation' are read "
            "by src.data.instances.read_protocol"
        ),
        "generator": "src/data/generator.py",
        "mode": args.mode,
        "seed": args.seed,
        "replicates": {str(k): v for k, v in sorted(args.replicates.items())},
        **(
            {"validation_per_size": args.validation_per_size}
            if args.validation_per_size
            else {"validation_fraction": args.validation_fraction}
        ),
        "validation_strategy": validation_strategy,
        "widen": bool(args.widen),
        "splits": {
            "train": sorted(train_entries),
            "validation": sorted(validation_entries),
        },
        "evaluation": {
            suite: paths
            for suite, paths in json.loads(args.splits.read_text()).get(
                "evaluation", {}
            ).items()
            if suite in EVALUATION_SUITES
        },
        "counts": {
            "train": len(train_entries),
            "validation": len(validation_entries),
            "cells": len(coordinates),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1))
    args.manifest.with_suffix(".specs.json").write_text(json.dumps(specs, indent=1))

    print(
        f"mode={args.mode} cells={len(coordinates)} "
        f"replicates={ {k: v for k, v in sorted(args.replicates.items())} }"
    )
    print(f"train={len(train_entries)} validation={len(validation_entries)}")
    print(f"wrote {args.output} and {args.manifest}")


if __name__ == "__main__":
    main()
