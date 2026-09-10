"""Protocol-driven single-project instance loading for the RL training stack.

Every training/evaluation entry point must consume ``splits.json`` (produced
by ``scripts/generate_pool.py``) and parse instances through the unified
``src.data.parsers`` parser + ``src.data.adapter`` bridge, so PPO shares exactly
the same data representation (and the same serial SGS decoder) as the
priority-rule / GA / GPHH baselines.

Instance *names* are the data-root-relative path without its extension (e.g.
``oras/RCPSP/RG30/Set 5/Pat80``).  File stems collide across the RG30 ``Set``
directories, and the PPO static graph cache requires unique names, so plain
``Path.stem`` is not usable here.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from src.core.rcpsp import Instance
from src.data.adapter import load_core_instance


def instance_id(rel: str | Path) -> str:
    """Unique instance name for one data-root-relative file path."""
    return str(Path(rel).with_suffix(""))


def load_single_project(root: str | Path, rel: str | Path) -> Instance:
    """Parse one single-project ``.sm``/``.rcp`` instance and adapt it."""
    path = Path(root) / Path(rel)
    return load_core_instance(path, name=instance_id(rel))


def loader_for(root: str | Path):
    """Return a picklable path -> Instance callable rooted at ``root``.

    ``functools.partial`` over the module-level ``load_single_project`` keeps
    the loader serializable for subproc vector environments (spawn).
    """
    return partial(load_single_project, Path(root))


def read_protocol(splits_path: str | Path) -> dict:
    """Read the fixed train/validation/evaluation protocol from ``splits.json``.

    Returns ``{"train": [...], "validation": [...], "evaluation": {suite: [...]}}``
    where every path is data-root-relative and posix-normalised.

    Split keys are ``train`` / ``validation``; the historical key names
    ``rg30_train`` / ``rg30_validation`` (when RG30 was the training pool) are
    still accepted so older manifests keep loading.
    """
    doc = json.loads(Path(splits_path).read_text())
    splits = doc["splits"]
    evaluation = doc["evaluation"]
    train = splits.get("train") or splits.get("rg30_train")
    validation = splits.get("validation") or splits.get("rg30_validation")
    if not train or not validation:
        raise ValueError(f"{splits_path}: missing train / validation splits")
    if not evaluation:
        raise ValueError(f"{splits_path}: evaluation groups are empty")
    norm = lambda paths: [Path(p).as_posix() for p in paths]
    return {
        "train": norm(train),
        "validation": norm(validation),
        "evaluation": {suite: norm(paths) for suite, paths in evaluation.items()},
    }
