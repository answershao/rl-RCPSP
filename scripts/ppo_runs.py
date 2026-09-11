"""Resolve timestamped PPO training runs.

CPU PPO runs are stored as ``cpu_YYYYMMDD_HHMMSS`` directories.  This module
keeps the selection rule shared by shell entry points and Python search code:
choose the newest run whose requested model file already exists.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CPU_RUN_NAME = re.compile(r"^cpu_\d{8}_\d{6}$")


def _model_path(run_dir: Path, model_file: Path) -> Path:
    if model_file.is_absolute():
        return model_file
    return run_dir / model_file


def latest_training_run(runs_root: Path, model_file: Path) -> Path:
    """Return the newest timestamped run containing ``model_file``.

    Directory names are used as the ordering source rather than filesystem
    mtime, because evaluation/search outputs can modify an older run later.
    """
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise FileNotFoundError(f"PPO runs directory not found: {runs_root}")
    if model_file.is_absolute():
        raise ValueError("model_file must be relative when selecting a run")

    candidates = sorted(
        (
            path
            for path in runs_root.iterdir()
            if path.is_dir() and CPU_RUN_NAME.fullmatch(path.name)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in candidates:
        if _model_path(run_dir, model_file).is_file():
            return run_dir

    expected = runs_root / "cpu_YYYYMMDD_HHMMSS" / model_file
    raise FileNotFoundError(
        f"no timestamped PPO run under {runs_root} contains {model_file}; "
        f"expected a model such as {expected}"
    )


def resolve_model_path(
    models_root: Path,
    model_file: Path,
    seed: int,
) -> Path:
    """Resolve a model from either legacy seed or timestamped-run layout.

    Supported layouts are:

    * ``<models-root>/seed<N>/<model-file>``;
    * ``<models-root>/<model-file>`` when ``models_root`` is one run; and
    * the newest ``<models-root>/cpu_YYYYMMDD_HHMMSS/<model-file>``.
    """
    models_root = Path(models_root)
    model_file = Path(model_file)
    if model_file.is_absolute():
        if not model_file.is_file():
            raise FileNotFoundError(f"PPO model not found: {model_file}")
        return model_file

    direct = models_root / model_file
    if direct.is_file():
        return direct

    seeded = models_root / f"seed{seed}" / model_file
    if seeded.is_file():
        return seeded

    run_dir = latest_training_run(models_root, model_file)
    return run_dir / model_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        help="resolve a model path, preserving legacy seedN layout support",
    )
    parser.add_argument(
        "--latest-run-dir",
        action="store_true",
        help="print the newest timestamped run directory containing the model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.latest_run_dir == (args.seed is not None):
        raise ValueError("specify exactly one of --latest-run-dir or --seed")
    if args.latest_run_dir:
        print(latest_training_run(args.models_root, args.model_file))
    else:
        print(resolve_model_path(args.models_root, args.model_file, args.seed))


if __name__ == "__main__":
    main()
