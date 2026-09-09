#!/usr/bin/env python3
"""Compare PPO makespans from two runs on an identical instance set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, dict[str, int]]:
    with path.open(newline="", encoding="ascii") as result_file:
        reader = csv.DictReader(result_file)
        required = {"instance", "PPO", "FIFO"}
        if missing := required - set(reader.fieldnames or ()):
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows: dict[str, dict[str, int]] = {}
        for raw in reader:
            instance = raw["instance"]
            if instance in rows:
                raise ValueError(f"{path} contains duplicate instance {instance!r}")
            rows[instance] = {"PPO": int(raw["PPO"]), "FIFO": int(raw["FIFO"])}
    if not rows:
        raise ValueError(f"{path} contains no result rows")
    return rows


def compare_results(reference: Path, candidate: Path, output: Path) -> Path:
    reference_rows = _load(reference)
    candidate_rows = _load(candidate)
    if reference_rows.keys() != candidate_rows.keys():
        raise ValueError("reference and candidate must cover the same instances")

    rows = []
    reference_gaps = []
    candidate_gaps = []
    for instance, reference_row in reference_rows.items():
        candidate_row = candidate_rows[instance]
        if reference_row["FIFO"] != candidate_row["FIFO"]:
            raise ValueError(f"FIFO result differs for {instance}")
        fifo = reference_row["FIFO"]
        if fifo <= 0:
            raise ValueError(f"FIFO makespan must be positive for {instance}")
        reference_gap = (reference_row["PPO"] - fifo) / fifo
        candidate_gap = (candidate_row["PPO"] - fifo) / fifo
        reference_gaps.append(reference_gap)
        candidate_gaps.append(candidate_gap)
        rows.append(
            {
                "instance": instance,
                "reference_PPO": reference_row["PPO"],
                "candidate_PPO": candidate_row["PPO"],
                "makespan_delta": candidate_row["PPO"] - reference_row["PPO"],
                "reference_FIFO_gap": f"{reference_gap:.6f}",
                "candidate_FIFO_gap": f"{candidate_gap:.6f}",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="ascii") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    deltas = np.asarray([row["makespan_delta"] for row in rows])
    reference_mean = np.mean([row["PPO"] for row in reference_rows.values()])
    candidate_mean = np.mean([row["PPO"] for row in candidate_rows.values()])
    print(
        "comparison: "
        f"reference_mean={reference_mean:.2f}; candidate_mean={candidate_mean:.2f}; "
        f"reference_FIFO_gap={np.mean(reference_gaps):+.6f}; "
        f"candidate_FIFO_gap={np.mean(candidate_gaps):+.6f}; "
        f"wins/ties/losses={np.sum(deltas < 0)}/{np.sum(deltas == 0)}/{np.sum(deltas > 0)}"
    )
    print(f"comparison details: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare_results(args.reference, args.candidate, args.output)


if __name__ == "__main__":
    main()
