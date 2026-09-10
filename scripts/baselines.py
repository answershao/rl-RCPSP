#!/usr/bin/env python3
"""Single-pass priority-rule baselines over the four single-project RCPSP suites.

Instances are discovered under ``--data-root``:

  PSPLIB    data/psplib/{j30,j60,j90,j120}/*.sm   (BKS known, main benchmark)
  generated data/generated/psp_grid_bal/**/*.rcp  (training reference rules)

Every instance is parsed by the unified parser (``src.data.parsers``), adapted
to the core ``Instance`` shape (``src.data.adapter``) and scheduled once per
(rule, scheme) pair.  Static rules run under both the serial and the parallel
SGS; WCS (Kolisch 1996) is parallel-only.  Output is an instance x method
makespan matrix in ``--output-csv``.

Runner plumbing (suite discovery, process pool, CSV writing) lives in
``scripts.common`` and is shared with ``scripts/run_ga.py`` /
``scripts/run_gphh.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import (
    add_instance_args,
    jobs_for_suites,
    map_jobs,
    print_suite_means,
    resolve_suite_ids,
    validate_instance_args,
    write_csv,
)
from src.core.rules import available_columns, all_makespans, column_name
from src.data.adapter import load_core_instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_instance_args(parser, workers_flag="--instance-workers")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/baselines_rcpsp/makespan_summary.csv"),
    )
    return parser.parse_args()


def evaluate_instance(
    path: Path, suite: str, seed: int, data_root: Path
) -> dict[str, float | int | str]:
    """Evaluate one instance (process-pool compatible)."""
    instance = load_core_instance(path)
    makespans = all_makespans(instance, seed=seed)
    return {
        "suite": suite,
        "instance": path.name,
        "file": str(path.relative_to(data_root)),
        "n_activities": len(instance.activities),
        "n_resources": instance.resource_count,
        **makespans,
    }


def print_result(row: dict[str, float | int | str]) -> None:
    serial = " ".join(
        f"{column_name('serial', rule)}={row[column_name('serial', rule)]}"
        for rule in ("FIFO", "LST", "LFT", "MTS", "GRPW")
    )
    parallel = (
        f"parallel_FIFO={row['parallel_FIFO']} parallel_LST={row['parallel_LST']} "
        f"parallel_WCS={row['parallel_WCS']}"
    )
    print(f"{row['suite']}/{row['instance']}: {serial} | {parallel}", flush=True)


def write_summary(
    rows: list[dict[str, float | int | str]],
    output_csv: Path,
    *,
    schemes: tuple[str, ...],
) -> Path:
    columns = available_columns(schemes)
    write_csv(
        rows,
        output_csv,
        fieldnames=["suite", "instance", "file", "n_activities", "n_resources", *columns],
    )
    print(f"makespan summary: {output_csv}")
    print_suite_means(
        rows,
        ("serial_FIFO", "serial_LST", "serial_LFT", "parallel_LST", "parallel_WCS"),
    )
    return output_csv


def main() -> None:
    args = parse_args()
    validate_instance_args(args.max_instances, args.instance_workers)
    suite_ids = resolve_suite_ids(args.suites)
    schemes = ("serial", "parallel")

    jobs = jobs_for_suites(
        args.data_root, suite_ids, args.max_instances, args.seed, args.data_root
    )
    rows = (
        map_jobs(
            evaluate_instance,
            jobs,
            workers=min(args.instance_workers, len(jobs)),
            on_result=print_result,
        )
        if jobs
        else []
    )
    write_summary(rows, args.output_csv, schemes=schemes)


if __name__ == "__main__":
    main()
