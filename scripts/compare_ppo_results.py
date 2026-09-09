#!/usr/bin/env python3
"""Compare PPO makespans from two runs on an identical instance set.

Both inputs are ``ppo_eval_summary.csv`` files as written by
``scripts/train_ppo.py`` (columns ``suite,file,n_activities,n_resources,
ppo_makespan``).  Instances are matched on the unique ``file`` column
(data-root-relative path), which stays stable across runs.

A rule reference (a ``baselines.py`` CSV + column such as ``serial_LST``) is
optional but recommended so each run can also be reported as a rule-relative
gap, mirroring the aggregate comparison table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.instances import instance_id


def _load(path: Path, column: str) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as result_file:
        reader = csv.DictReader(result_file)
        if "file" not in (reader.fieldnames or ()):
            raise ValueError(f"{path} is missing the 'file' column")
        if column not in (reader.fieldnames or ()):
            raise ValueError(f"{path} is missing column {column!r}")
        rows: dict[str, int] = {}
        for raw in reader:
            key = instance_id(raw["file"])
            if key in rows:
                raise ValueError(f"{path} contains duplicate instance {key!r}")
            rows[key] = int(float(raw[column]))
    if not rows:
        raise ValueError(f"{path} contains no result rows")
    return rows


def _load_reference(path: Path, column: str) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if column not in (reader.fieldnames or ()):
            raise ValueError(f"{path} is missing reference column {column!r}")
        refs: dict[str, int] = {}
        for raw in reader:
            key = instance_id(raw["file"])
            try:
                refs[key] = int(float(raw[column]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid {column} for {raw.get('file')!r}") from exc
    return refs


def compare_results(
    reference: Path,
    candidate: Path,
    output: Path,
    reference_rules: dict[str, int] | None = None,
) -> Path:
    reference_rows = _load(reference, "ppo_makespan")
    candidate_rows = _load(candidate, "ppo_makespan")
    if reference_rows.keys() != candidate_rows.keys():
        missing = sorted(set(reference_rows) - set(candidate_rows))
        extra = sorted(set(candidate_rows) - set(reference_rows))
        raise ValueError(
            "reference and candidate must cover the same instances; "
            f"missing in candidate: {len(missing)}; extra in candidate: {len(extra)}"
        )

    rows = []
    reference_gaps: list[float] = []
    candidate_gaps: list[float] = []
    for instance, reference_makespan in reference_rows.items():
        candidate_makespan = candidate_rows[instance]
        row: dict[str, float | int | str] = {
            "instance": instance,
            "reference_PPO": reference_makespan,
            "candidate_PPO": candidate_makespan,
            "makespan_delta": candidate_makespan - reference_makespan,
        }
        if reference_rules:
            rule_value = reference_rules.get(instance)
            if rule_value is None:
                raise ValueError(f"reference rules do not cover {instance}")
            if rule_value <= 0:
                raise ValueError(f"reference rule makespan must be positive for {instance}")
            reference_gap = (reference_makespan - rule_value) / rule_value
            candidate_gap = (candidate_makespan - rule_value) / rule_value
            row["reference_rule_gap"] = f"{reference_gap:.6f}"
            row["candidate_rule_gap"] = f"{candidate_gap:.6f}"
            reference_gaps.append(reference_gap)
            candidate_gaps.append(candidate_gap)
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    deltas = np.asarray([row["makespan_delta"] for row in rows], dtype=np.float64)
    reference_mean = float(np.mean([row["reference_PPO"] for row in rows]))
    candidate_mean = float(np.mean([row["candidate_PPO"] for row in rows]))
    print(
        "comparison: "
        f"reference_mean={reference_mean:.2f}; candidate_mean={candidate_mean:.2f}; "
        f"wins/ties/losses={int((deltas < 0).sum())}/"
        f"{int((deltas == 0).sum())}/{int((deltas > 0).sum())}"
    )
    if reference_rules:
        print(
            f"rule-relative gap: reference={np.mean(reference_gaps):+.6f}; "
            f"candidate={np.mean(candidate_gaps):+.6f}"
        )
    print(f"comparison details: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="baseline run ppo_eval_summary.csv")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="candidate run ppo_eval_summary.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ref-rules", type=Path, default=None,
        help="baselines.py CSV covering the instances (optional)",
    )
    parser.add_argument(
        "--ref-rule", default="serial_LST",
        help="column of --ref-rules used for the relative gap (default: serial_LST)",
    )
    args = parser.parse_args()
    reference_rules = (
        _load_reference(args.ref_rules, args.ref_rule) if args.ref_rules else None
    )
    compare_results(args.reference, args.candidate, args.output, reference_rules)


if __name__ == "__main__":
    main()
