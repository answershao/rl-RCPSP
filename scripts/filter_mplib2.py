#!/usr/bin/env python3
"""Copy MPLIB2 instances matching the requested project/activity/resource size.

Examples:
    python3 scripts/filter_mplib2.py data/MPLIB2
    python3 scripts/filter_mplib2.py data/MPLIB2 --projects 10 --activities-per-project 50 --resources 5
    python3 scripts/filter_mplib2.py data/MPLIB2 --output-dir data/MPLIB2_train_10_50_5 --link
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any


VIRTUAL_ACTIVITIES_PER_PROJECT = 2


def parse_summary(path: Path) -> list[dict[str, Any]]:
    """Read an MPLIB2 summary CSV and normalize numeric fields."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if not reader.fieldnames:
            raise ValueError(f"empty CSV header: {path}")
        for raw in reader:
            row: dict[str, Any] = {}
            for field, value in raw.items():
                if field is None:
                    continue
                value = value.strip().replace(",", ".")
                if not value:
                    parsed: Any = ""
                else:
                    try:
                        number = float(value)
                    except ValueError:
                        parsed = value
                    else:
                        parsed = int(number) if number.is_integer() else number
                row[field] = parsed
            rows.append(row)
    return rows


def select_rows(
    summary: Path, *, projects: int, resources: int
) -> list[dict[str, Any]]:
    rows = parse_summary(summary)
    return [
        row
        for row in rows
        if (
            int(row["J"]) == projects
            and int(row["K"]) == resources
        )
    ]


def project_activity_counts(path: Path) -> tuple[int, ...]:
    """Read the activity count recorded for each project in an RCMP file."""
    nonempty = (line.split() for line in path.open(encoding="utf-8") if line.split())
    project_count = int(next(nonempty)[0])
    resource_count = int(next(nonempty)[0])
    next(nonempty)  # resource capacities

    counts = []
    for _ in range(project_count):
        activity_count = int(next(nonempty)[0])
        counts.append(activity_count)
        next(nonempty)  # project availability vector
        for _ in range(activity_count):
            next(nonempty)
    if resource_count < 1:
        raise ValueError(f"{path}: invalid RCMP project structure")
    return tuple(counts)


def link_or_copy(source: Path, destination: Path, *, use_hard_link: bool) -> None:
    if destination.exists():
        destination.unlink()
    if use_hard_link:
        destination.hardlink_to(source)
    else:
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("data/MPLIB2"))
    parser.add_argument("--projects", type=int, default=10)
    parser.add_argument(
        "--activities-per-project",
        type=int,
        default=50,
        help="number of activities per project (default: 50)",
    )
    parser.add_argument("--resources", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory (default: data/MPLIB2_train_<projects>_<activities-per-project>_<resources>)",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="create hard links instead of copying files (requires the same filesystem)",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"root does not exist: {args.root}")

    if args.projects < 1 or args.activities_per_project < 1 or args.resources < 1:
        parser.error("projects, activities-per-project, and resources must be positive")

    output_dir = args.output_dir or Path(
        f"data/MPLIB2_train_{args.projects}_{args.activities_per_project}_{args.resources}"
    )
    stored_activities_per_project = (
        args.activities_per_project + VIRTUAL_ACTIVITIES_PER_PROJECT
    )

    summaries = sorted(args.root.glob("MPLIB 2 - Set */Summary_Set_*.csv"))
    if not summaries:
        parser.error(f"no MPLIB2 Summary_Set_*.csv files found under {args.root}")

    selected: list[dict[str, Any]] = []
    missing: list[tuple[Path, Path]] = []
    for summary in summaries:
        set_dir = summary.parent
        instance_dir = set_dir / "Instances"
        set_number = summary.name.rsplit("_", 1)[-1].split(".", 1)[0]
        for row in select_rows(
            summary,
            projects=args.projects,
            resources=args.resources,
        ):
            source = instance_dir / f"MPLIB2_Set{set_number}_{row['Instance']}.rcmp"
            if not source.is_file():
                missing.append((summary, source))
                continue
            if any(
                count != stored_activities_per_project
                for count in project_activity_counts(source)
            ):
                continue
            row["set"] = set_dir.name
            row["source"] = str(source)
            row["destination"] = ""
            selected.append(row)

    if missing:
        print(
            f"error: {len(missing)} summary rows have no matching .rcmp file; "
            f"first example: {missing[0][1]}",
            file=sys.stderr,
        )
        return 2

    if not selected:
        print(
            f"no instances with J={args.projects}, "
            f"{args.activities_per_project} activities/project, "
            f"K={args.resources} found"
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    destinations: set[str] = set()
    for row in selected:
        destination = output_dir / Path(row["source"]).name
        if destination.name in destinations:
            print(
                f"error: duplicate destination filename {destination.name}",
                file=sys.stderr,
            )
            return 2
        destinations.add(destination.name)
        link_or_copy(Path(row["source"]), destination, use_hard_link=args.link)
        row["destination"] = str(destination)

    manifest = output_dir / "instances.csv"
    summary_fields = [
        field for field in selected[0] if field not in {"source", "destination"}
    ]
    fields = [*summary_fields, "source", "destination"]
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in selected)

    mode = "hard links" if args.link else "copies"
    print(
        f"selected {len(selected)} instances (J={args.projects}, "
        f"{args.activities_per_project} activities/project, "
        f"K={args.resources}) as {mode} in {output_dir}"
    )
    print(f"manifest: {manifest}")
    print(f"total matching instances: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
