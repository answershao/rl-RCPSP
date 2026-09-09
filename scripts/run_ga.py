#!/usr/bin/env python3
"""Random-key genetic algorithm baseline over the single-project RCPSP suites.

Instances are discovered under ``--data-root`` exactly like the priority-rule
runner ``scripts/baselines.py`` (same suite ids, same natural ordering), and
each chromosome is decoded by the *same* serial SGS, so the resulting makespans
are directly comparable to the ``serial_*`` rule columns.

Every instance is parsed by the unified parser, adapted to the core ``Instance``
shape and optimised with ``src.core.ga.run_ga``.  Output is one row per instance
with ``ga_makespan`` / ``ga_best_generation``; rows can be merged with the rule
matrix on ``(suite, instance, file)``.

Example (j30, 40 workers, default budget 50 x 200):
    python -m scripts.run_ga --data-root data --suites psplib_j30 \
        --instance-workers 40 --output-csv outputs/ga_rcpsp/ga_j30.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import zlib

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
from src.core.ga import DEFAULT_ELITE, DEFAULT_GENERATIONS, DEFAULT_POPULATION, GAParameters, run_ga
from src.data.adapter import load_core_instance

GA_FIELDNAMES = [
    "suite",
    "instance",
    "file",
    "n_activities",
    "n_resources",
    "ga_makespan",
    "ga_best_generation",
    "ga_evaluations",
    "ga_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_instance_args(parser, workers_flag="--instance-workers")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/ga_rcpsp/ga_makespan_summary.csv"),
    )
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--tournament", type=int, default=3)
    parser.add_argument("--elite", type=int, default=DEFAULT_ELITE)
    parser.add_argument(
        "--mutation",
        type=float,
        default=None,
        help="per-gene mutation probability; default 1 / n_activities",
    )
    return parser.parse_args()


def evaluate_instance(
    path: Path,
    suite: str,
    seed: int,
    data_root: Path,
    params: GAParameters,
) -> dict[str, float | int | str]:
    instance = load_core_instance(path)
    # Derive a stable per-instance seed so equal-sized instances (e.g. all j30)
    # do not share the same initial population, while remaining reproducible.
    file_rel = str(path.relative_to(data_root))
    instance_seed = seed ^ zlib.crc32(file_rel.encode("utf-8"))
    started = time.perf_counter()
    result = run_ga(instance, seed=instance_seed, parameters=params)
    elapsed = time.perf_counter() - started
    return {
        "suite": suite,
        "instance": path.name,
        "file": str(path.relative_to(data_root)),
        "n_activities": len(instance.activities),
        "n_resources": instance.resource_count,
        "ga_makespan": result.best_makespan,
        "ga_best_generation": result.best_generation,
        "ga_evaluations": result.evaluations,
        "ga_seconds": round(elapsed, 3),
    }


def print_result(row: dict[str, float | int | str]) -> None:
    print(
        f"{row['suite']}/{row['instance']}: ga={row['ga_makespan']} "
        f"(gen {row['ga_best_generation']}, {row['ga_evaluations']} evals, "
        f"{row['ga_seconds']}s)",
        flush=True,
    )


def write_summary(
    rows: list[dict[str, float | int | str]], output_csv: Path
) -> Path:
    write_csv(rows, output_csv, fieldnames=GA_FIELDNAMES)
    print(f"GA makespan summary: {output_csv}")
    print_suite_means(rows, ("ga_makespan",), prefix="mean ")
    return output_csv


def main() -> None:
    args = parse_args()
    validate_instance_args(args.max_instances, args.instance_workers)
    suite_ids = resolve_suite_ids(args.suites)
    params = GAParameters(
        population=args.population,
        generations=args.generations,
        tournament=args.tournament,
        elite=args.elite,
        mutation=args.mutation,
    )

    jobs = jobs_for_suites(
        args.data_root, suite_ids, args.max_instances, args.seed, args.data_root, params
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
    write_summary(rows, args.output_csv)


if __name__ == "__main__":
    main()
