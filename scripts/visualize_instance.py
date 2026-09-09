#!/usr/bin/env python3
"""Render the Gantt chart and AON network for one RCMP instance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.rcmpsp import generate_schedule, parse_rcmp, priority_fifo
from src.visualization.aon import plot_aon
from src.visualization.gantt import plot_gantt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path, help="path to one .rcmp instance")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for the two PNG files (default: outputs/visualizations/<instance>)",
    )
    return parser.parse_args()


def render_instance(instance_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Create both visualizations for one instance and return their paths."""
    instance_path = instance_path.expanduser()
    if not instance_path.is_file():
        raise FileNotFoundError(f"instance file does not exist: {instance_path}")

    instance = parse_rcmp(instance_path)
    schedule = generate_schedule(instance, priority_fifo)
    output_dir = output_dir or Path("outputs") / "visualizations" / instance_path.stem

    gantt_path = plot_gantt(instance, schedule, output_dir / "gantt.png")
    aon_path = plot_aon(instance, output_dir / "aon.png", schedule=schedule)
    return gantt_path, aon_path


def main() -> None:
    args = parse_args()
    gantt_path, aon_path = render_instance(args.instance, args.output_dir)
    print(f"output directory: {gantt_path.parent}")
    print(f"gantt: {gantt_path}")
    print(f"aon: {aon_path}")


if __name__ == "__main__":
    main()
