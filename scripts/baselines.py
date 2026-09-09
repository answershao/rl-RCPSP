#!/usr/bin/env python3
"""Evaluate scheduling baselines on the MPLIB2 10-project/500-activity training set."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from multiprocessing import get_context
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.core.exact import solve_exact
from src.core.rcmpsp import baseline_makespans, parse_rcmp


_INSTANCE_NAME_PATTERN = re.compile(
    r"^MPLIB2_Set(?P<set_number>\d+)_(?P<instance_number>\d+)\.rcmp$",
    re.IGNORECASE,
)


def instance_sort_key(path: Path) -> tuple[int, int, int, str]:
    """Sort MPLIB2 files by numeric set and instance numbers."""
    match = _INSTANCE_NAME_PATTERN.fullmatch(path.name)
    if match is None:
        # Keep unexpected names deterministic, after the standard names.
        return (1, 0, 0, path.name)
    return (
        0,
        int(match.group("set_number")),
        int(match.group("instance_number")),
        path.name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=Path("data/MPLIB2_train_10_50_5"),
        help="directory containing the .rcmp baseline instances",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "outputs/baselines/runs/mplib2_10_50_5/makespan_summary.csv"
        ),
        help=(
            "path for a new baseline matrix; the fixed reference remains at "
            "outputs/baselines_mplib2_10_50_5/makespan_summary.csv"
        ),
    )
    parser.add_argument("--seed", type=int, default=7, help="seed for the random-priority baseline")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="evaluate at most this many instances; 0 evaluates all",
    )
    parser.add_argument(
        "--exact-time-limit",
        type=float,
        default=0.0,
        help="CP-SAT seconds per instance; use 0 to omit the CP-SAT column.",
    )
    parser.add_argument("--exact-workers", type=int, default=1, help="CP-SAT workers; 1 is reproducible.")
    parser.add_argument(
        "--instance-workers",
        type=int,
        default=1,
        help="number of instances to solve concurrently in separate processes",
    )
    return parser.parse_args()


def find_instances(root: Path) -> list[Path]:
    """Return every RCMP file in a prepared instance directory."""
    paths = sorted(root.glob("*.rcmp"), key=instance_sort_key)
    if not paths:
        raise ValueError(f"no .rcmp instances found under {root}")
    return paths


def evaluate_instance(
    path: Path, seed: int, exact_time_limit: float, exact_workers: int
) -> tuple[dict[str, float | int | str], str]:
    """Evaluate one instance in a process-pool-compatible function."""
    instance = parse_rcmp(path)
    baselines = baseline_makespans(instance, seed)
    row: dict[str, float | int | str] = {
        "instance": path.name,
        "FIFO": baselines["fifo"],
        "Shortest": baselines["shortest"],
        "Random": baselines["random"],
    }
    exact_details = ""
    if exact_time_limit:
        exact = solve_exact(
            instance, time_limit=exact_time_limit, workers=exact_workers
        )
        row["CP-SAT"] = exact.schedule.makespan
        row["Remark"] = exact.status.lower()
        row["Bound"] = exact.best_bound
        row["Wall Time"] = exact.wall_time
        exact_details = (
            f" CP-SAT={row['CP-SAT']} ({exact.status}; "
            f"bound={exact.best_bound:.0f}; {exact.wall_time:.2f}s)"
        )
    return row, exact_details


def print_result(row: dict[str, float | int | str], exact_details: str) -> None:
    print(
        f"{row['instance']}: FIFO={row['FIFO']} Shortest={row['Shortest']} "
        f"Random={row['Random']}" + exact_details,
        flush=True,
    )


def evaluate_instances(
    paths: list[Path], *, seed: int, exact_time_limit: float,
    exact_workers: int, instance_workers: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate instances concurrently while retaining input order."""
    if instance_workers == 1:
        rows = []
        for path in paths:
            row, exact_details = evaluate_instance(
                path, seed, exact_time_limit, exact_workers
            )
            print_result(row, exact_details)
            rows.append(row)
        return rows

    ordered_rows: list[dict[str, float | int | str] | None] = [None] * len(paths)
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=instance_workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                evaluate_instance, path, seed, exact_time_limit, exact_workers
            ): index
            for index, path in enumerate(paths)
        }
        for future in as_completed(futures):
            row, exact_details = future.result()
            ordered_rows[futures[future]] = row
            print_result(row, exact_details)

    if any(row is None for row in ordered_rows):
        raise RuntimeError("one or more instance evaluations did not return a result")
    return [row for row in ordered_rows if row is not None]


def write_summary(
    rows: list[dict[str, float | int | str]], output_csv: Path, *, include_exact: bool
) -> Path:
    """Write the instance-by-method makespan matrix."""
    methods = ["FIFO", "Shortest", "Random"]
    metadata = []
    if include_exact:
        methods.append("CP-SAT")
        metadata = ["Remark", "Bound", "Wall Time"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="ascii") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=["instance", *methods, *metadata]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"makespan summary: {output_csv}")
    print(
        "mean: " + " ".join(
            f"{method}={np.mean([row[method] for row in rows]):.2f}"
            for method in methods
        )
    )
    return output_csv


def main() -> None:
    args = parse_args()
    if args.exact_time_limit < 0:
        raise ValueError("--exact-time-limit must be non-negative")
    if args.exact_workers < 1:
        raise ValueError("--exact-workers must be positive")
    if args.instance_workers < 1:
        raise ValueError("--instance-workers must be positive")
    if args.max_instances < 0:
        raise ValueError("--max-instances must be non-negative")
    paths = find_instances(args.instances_root)
    if args.max_instances:
        paths = paths[:args.max_instances]
    rows = evaluate_instances(
        paths,
        seed=args.seed,
        exact_time_limit=args.exact_time_limit,
        exact_workers=args.exact_workers,
        instance_workers=min(args.instance_workers, len(paths)),
    )
    write_summary(rows, args.output_csv, include_exact=bool(args.exact_time_limit))


if __name__ == "__main__":
    main()
