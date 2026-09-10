"""Shared runner machinery for the instance-matrix baseline scripts.

Single source of truth for the RCPSP suite matrix (benchmark suites *and*
training pools):

* ``SUITE_SPECS`` -- suite id -> (data-root-relative directory, glob, role).
  ``scripts/generate_pool.py`` (protocol manifest) and the instance runners
  (``scripts/baselines.py``, ``scripts/run_ga.py``, ``scripts/run_gphh.py``)
  all consume this table, so directory layout / suite ids / roles cannot drift.
* ``EVALUATION_SUITES`` / ``DEFAULT_SUITES`` -- which suites may be reported as
  benchmark results, and what the runners default to.  See the comments below.
* instance discovery (``find_instances`` with numeric-aware ordering), plus
  path helpers used when emitting data-root-relative ``file`` keys.
* the spawn-safe evaluation driver ``map_jobs`` (serial fallback for
  ``workers == 1``) shared by every runner.
* CSV / per-suite summary writers so all runners produce the same file layout
  and console shape.

This module must stay free of imports from ``src.*`` so it can be imported by
any script without pulling the whole learning stack.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from multiprocessing import get_context
from pathlib import Path
import re

import numpy as np

# suite id -> (relative directory under data-root, glob pattern, protocol role)
SUITE_SPECS: dict[str, tuple[str, str, str]] = {
    "psplib_j30": ("psplib/j30", "*.sm", "final-evaluation"),
    "psplib_j60": ("psplib/j60", "*.sm", "final-evaluation"),
    "psplib_j90": ("psplib/j90", "*.sm", "final-evaluation"),
    "psplib_j120": ("psplib/j120", "*.sm", "final-evaluation"),
    "rg30": ("oras/RCPSP/RG30", "**/*.rcp", "historical-training-pool"),
    "rg300": ("oras/RCPSP/RG300", "*.rcp", "final-evaluation"),
    "patterson": ("oras/RCPSP/Patterson", "*.rcp", "final-evaluation"),
    # The active training pool is registered here for one reason: S5's
    # checkpoint selection needs a reference-rule CSV covering the `validation`
    # split, and `scripts.baselines` is the only producer of that CSV -- it can
    # only find instances through this table.  See docs/EXECUTION_FLOW.md S3.
    "psp_grid": ("generated/psp_grid_bal", "**/*.rcp", "training-pool"),
}

# Suites that may be reported as benchmark results.  Every training pool is
# excluded: scoring a method on the data it was fitted to is not a benchmark.
# (rg30 is a retired pool, psp_grid is the active one.)
EVALUATION_SUITES = tuple(
    suite for suite, spec in SUITE_SPECS.items() if spec[2] == "final-evaluation"
)

# Default `--suites` for the baseline runners.  Pointing it at the training pool
# means the command S5 depends on needs no extra flags:
#   python -m scripts.baselines --instance-workers 8 \
#     --output-csv outputs/rules_psp_grid/makespan_summary.csv
# Benchmark runs state their suites explicitly, e.g.
#   --suites psplib_j30,psplib_j60,psplib_j90,psplib_j120
DEFAULT_SUITES = "psp_grid"

_NUMBER_TOKEN = re.compile(r"\d+")


def natural_key(path: Path) -> tuple:
    """Numeric-aware sort key so j3021_1 < j3021_10 and Pat2 < Pat10."""
    return tuple(int(token) if token.isdigit() else token for token in _NUMBER_TOKEN.split(path.stem)) + (
        path.name,
    )


def find_instances(root: Path, suite: str) -> list[Path]:
    """Return every instance of ``suite`` under ``root`` in natural order.

    Raises ``ValueError`` when the suite directory is missing or empty, so a
    mistyped layout fails loudly instead of silently writing an empty matrix.
    """
    directory, pattern, _ = SUITE_SPECS[suite]
    base = root / directory
    if not base.is_dir():
        raise ValueError(f"suite directory not found: {base}")
    paths = sorted(base.glob(pattern), key=natural_key)
    if not paths:
        raise ValueError(f"no instance files found for suite {suite!r} under {base}")
    return paths


def relative_posix(root: Path, path: Path) -> str:
    """Data-root-relative POSIX path used as the stable per-instance ``file`` key."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_suite_ids(raw: str) -> list[str]:
    """Split a comma-separated ``--suites`` value and validate every id."""
    suite_ids = [suite.strip() for suite in raw.split(",") if suite.strip()]
    unknown = [suite for suite in suite_ids if suite not in SUITE_SPECS]
    if unknown:
        raise ValueError(f"unknown suite ids: {unknown}; choose from {sorted(SUITE_SPECS)}")
    return suite_ids


def add_instance_args(parser: argparse.ArgumentParser, *, workers_flag: str) -> None:
    """Add the shared ``--data-root/--suites/--max-instances/--workers`` options."""
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--suites",
        default=DEFAULT_SUITES,
        help="comma-separated suite ids; defaults to psp_grid (the active "
        "training pool, used to build the reference-rule CSV). Pass the "
        "evaluation suites explicitly for benchmark runs.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="evaluate at most this many instances per suite; 0 evaluates all",
    )
    parser.add_argument(
        workers_flag,
        type=int,
        default=1,
        help="number of instances to evaluate concurrently in separate processes",
    )
    parser.add_argument("--seed", type=int, default=17, help="random seed for this run")


def validate_instance_args(max_instances: int, workers: int) -> None:
    if max_instances < 0:
        raise ValueError("--max-instances must be non-negative")
    if workers < 1:
        raise ValueError("instance workers must be positive")


def jobs_for_suites(
    data_root: Path,
    suite_ids: list[str],
    max_instances: int,
    *worker_args: object,
) -> list[tuple[Path, str, object, ...]]:
    """Flatten ``(path, suite, *worker_args)`` jobs over the requested suites.

    Suite order and per-suite natural order are preserved; ``max_instances``
    truncates each suite independently (0 = all).
    """
    jobs: list[tuple[Path, str, object, ...]] = []
    for suite in suite_ids:
        paths = find_instances(data_root, suite)
        if max_instances:
            paths = paths[:max_instances]
        jobs.extend((path, suite, *worker_args) for path in paths)
    return jobs


def map_jobs(
    worker: object,
    jobs: list[tuple],
    workers: int,
    *,
    on_result: object | None = None,
) -> list[dict[str, float | int | str]]:
    """Run ``worker(*job) -> row dict`` over ``jobs``, preserving input order.

    ``worker`` must be a picklable module-level callable (spawn pool).  With
    ``workers == 1`` (or a single job) everything runs in-process; otherwise a
    ``spawn`` ``ProcessPoolExecutor`` evaluates concurrently and rows are
    re-ordered into input order.  ``on_result(row)`` is invoked as each row
    completes (progress printing).
    """
    if workers == 1 or len(jobs) == 1:
        rows: list[dict[str, float | int | str]] = []
        for job in jobs:
            row = worker(*job)
            if on_result is not None:
                on_result(row)
            rows.append(row)
        return rows

    ordered: list[dict[str, float | int | str] | None] = [None] * len(jobs)
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(worker, *job): index for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            row = future.result()
            ordered[futures[future]] = row
            if on_result is not None:
                on_result(row)
    if any(row is None for row in ordered):
        raise RuntimeError("one or more instance evaluations did not return a result")
    return [row for row in ordered if row is not None]


def write_csv(
    rows: list[dict[str, float | int | str]],
    output_csv: Path,
    fieldnames: list[str],
) -> None:
    """Write a header + rows matrix, creating parent directories as needed."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_by_suite(rows: list[dict[str, float | int | str]]) -> dict[str, list[dict[str, float | int | str]]]:
    """Bucket rows by their ``suite`` value, preserving first-seen ordering."""
    by_suite: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_suite.setdefault(str(row["suite"]), []).append(row)
    return by_suite


def print_suite_means(
    rows: list[dict[str, float | int | str]],
    columns: tuple[str, ...],
    *,
    prefix: str = "",
) -> None:
    """Print ``suite (n=..): col=mean`` lines for every column, one per suite."""
    for suite in sorted(group_by_suite(rows)):
        suite_rows = group_by_suite(rows)[suite]
        means = " ".join(
            f"{column}={np.mean([row[column] for row in suite_rows]):.2f}"
            for column in columns
        )
        print(f"{suite} (n={len(suite_rows)}): {prefix}{means}")
