"""Fixed train/validation/test splits and a padded multi-instance environment."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.rcmpsp import parse_rcmp
from src.environments.rcmpsp_env import INVALID_ACTION_PENALTY, RCMPSPEnv
from src.environments.observation import (
    MAX_SUCCESSORS,
    flatten_observation,
    observation_size,
)

DEFAULT_INSTANCES_ROOT = Path("data/MPLIB2_train_10_50_5")
SPLIT_NAMES = ("train", "validation", "test")
MPLIB2_GROUP_FIELDS = (
    "set",
    "J",
    "I",
    "K",
    "a_I",
    "SP",
    "a_SP",
    "MP",
    "MF",
    "CR",
    "RD",
    "PD",
    "RS",
    "a_RS",
)


def make_splits(root: str | Path = DEFAULT_INSTANCES_ROOT) -> dict[str, list[str]]:
    """Split every five-replicate MPLIB2 parameter group into 3/1/1 sets."""
    root = Path(root)
    files = {path.name: path for path in root.glob("*.rcmp")}
    if not files:
        raise ValueError(f"no .rcmp instances found under {root}")
    manifest = root / "instances.csv"
    if not manifest.is_file():
        raise ValueError(f"MPLIB2 instance manifest not found: {manifest}")

    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        required_fields = {"Instance", "destination", *MPLIB2_GROUP_FIELDS}
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(
                f"{manifest} is missing required fields: {sorted(missing_fields)}"
            )
        rows = list(reader)

    groups: dict[tuple[str, ...], list[tuple[int, Path]]] = defaultdict(list)
    manifest_files = set()
    for row in rows:
        filename = Path(row["destination"]).name
        if filename in manifest_files:
            raise ValueError(f"{manifest} contains duplicate instance {filename!r}")
        if filename not in files:
            raise ValueError(f"manifest instance not found under {root}: {filename}")
        try:
            instance_number = int(row["Instance"])
        except ValueError as exc:
            raise ValueError(
                f"{manifest} contains invalid Instance value {row['Instance']!r}"
            ) from exc
        manifest_files.add(filename)
        group_key = tuple(row[field] for field in MPLIB2_GROUP_FIELDS)
        groups[group_key].append((instance_number, files[filename]))

    unlisted_files = sorted(files.keys() - manifest_files)
    if unlisted_files:
        raise ValueError(
            f"{manifest} does not list {len(unlisted_files)} RCMP files; "
            f"first missing file: {unlisted_files[0]}"
        )

    splits = {name: [] for name in SPLIT_NAMES}
    for group_key in sorted(groups):
        group = sorted(groups[group_key])
        if len(group) != 5:
            parameters = dict(zip(MPLIB2_GROUP_FIELDS, group_key))
            raise ValueError(
                "expected five replicate instances per MPLIB2 parameter group, "
                f"got {len(group)} for {parameters}"
            )
        paths = [str(path) for _, path in group]
        splits["train"].extend(paths[:3])
        splits["validation"].append(paths[3])
        splits["test"].append(paths[4])
    return splits


def make_overfit_splits(
    splits: dict[str, list[str]], count: int, *, seed: int
) -> dict[str, list[str]]:
    """Select one training replicate from each of ``count`` parameter groups.

    ``make_splits`` stores the three training replicates for every MPLIB2 group
    contiguously. Sampling groups instead of individual files prevents the
    diagnostic subset from being dominated by replicate copies of one setting.
    The selected instances are also used for validation so this mode measures
    whether PPO can fit its training instances, rather than generalization.
    """
    train_paths = splits.get("train", [])
    if len(train_paths) % 3:
        raise ValueError("training split must contain three replicates per parameter group")
    group_count = len(train_paths) // 3
    if not 1 <= count <= group_count:
        raise ValueError(f"count must be between 1 and {group_count}")

    rng = random.Random(seed)
    selected_groups = sorted(rng.sample(range(group_count), count))
    selected = [
        train_paths[3 * group_index + rng.randrange(3)]
        for group_index in selected_groups
    ]
    return {
        "train": selected,
        "validation": list(selected),
        "test": [],
    }


def write_split_manifest(splits: dict[str, list[str]], path: str | Path) -> Path:
    """Write an already resolved split manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n")
    return path


def write_splits(
    path: str | Path = "splits.json",
    root: str | Path = DEFAULT_INSTANCES_ROOT,
) -> Path:
    return write_split_manifest(make_splits(root), path)


def partition_instance_catalog(
    paths: Sequence[str], parts: int
) -> list[tuple[list[str], list[int]]]:
    """Split a training catalog into per-worker subsets with global indices."""
    if parts < 1:
        raise ValueError("parts must be positive")
    if not paths:
        raise ValueError("paths must not be empty")
    if parts > len(paths):
        raise ValueError("parts cannot exceed the number of paths")
    return [
        (
            [str(paths[index]) for index in range(part, len(paths), parts)],
            list(range(part, len(paths), parts)),
        )
        for part in range(parts)
    ]


class MultiInstanceRCMPSPEnv(gym.Env[np.ndarray, int]):
    """Sample one instance per episode with fixed padded observation and action spaces."""

    def __init__(self, instances: list[str | Path],
                 max_activities: int | None = None, max_resources: int | None = None,
                 instance_indices: list[int] | None = None,
                 catalog_size: int | None = None):
        super().__init__()
        if not instances:
            raise ValueError("instances must not be empty")
        self.instance_paths = [Path(path) for path in instances]
        self.instances = [parse_rcmp(path) for path in self.instance_paths]
        self.instance_indices = (
            list(range(len(self.instances))) if instance_indices is None else instance_indices
        )
        self.catalog_size = len(self.instances) if catalog_size is None else catalog_size
        if len(self.instance_indices) != len(self.instances):
            raise ValueError("instance_indices must match instances")
        if any(not 0 <= index < self.catalog_size for index in self.instance_indices):
            raise ValueError("instance_indices must refer to the static graph catalog")
        self.max_activities = max([max_activities or 0] + [len(instance.activities) for instance in self.instances])
        self.max_resources = max([max_resources or 0] + [instance.resource_count for instance in self.instances])
        self.max_successors = MAX_SUCCESSORS
        self.action_space = spaces.Discrete(self.max_activities)
        feature_size = observation_size(self.max_activities, self.max_resources)
        self.observation_space = spaces.Box(0.0, 1.0, (feature_size,), dtype=np.float32)
        # Reuse environment objects across episodes; reset only clears mutable
        # scheduling state and avoids repeated allocation of large buffers.
        self._envs = [RCMPSPEnv(instance) for instance in self.instances]
        self._env: RCMPSPEnv | None = None
        self._flat_buffers: dict[int, np.ndarray] = {}
        self._capacity_scales = {
            index: np.maximum(np.asarray(instance.capacities, dtype=np.float32), 1.0)
            for index, instance in enumerate(self.instances)
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = int(self.np_random.integers(len(self.instances)))
        self._env = self._envs[index]
        observation, info = self._env.reset(seed=seed)
        self._active_index = index
        return self._encode(observation), {**info, "instance": self.instance_paths[index].name}

    def step(self, action):
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.max_activities})")
        action = int(action)
        if action >= self.active_env.activity_count:
            observation = self.active_env._observation()
            return (
                self._encode(observation),
                INVALID_ACTION_PENALTY,
                False,
                False,
                {
                    "makespan": self.active_env._state.current_time,
                    "eligible_mask": observation["eligible_mask"].copy(),
                    "invalid_action": True,
                },
            )
        observation, reward, terminated, truncated, info = self._env.step(action)
        return self._encode(observation), reward, terminated, truncated, info

    @property
    def active_env(self) -> RCMPSPEnv:
        if self._env is None:
            raise RuntimeError("environment has not been reset")
        return self._env

    def _encode(self, observation):
        env = self.active_env
        buffer = self._flat_buffers.get(self._active_index)
        if buffer is None:
            buffer = np.zeros(
                observation_size(self.max_activities, self.max_resources), dtype=np.float32
            )
            self._flat_buffers[self._active_index] = buffer
        return flatten_observation(
            observation,
            env.instance.capacities,
            env.horizon,
            instance_index=self.instance_indices[self._active_index],
            catalog_size=self.catalog_size,
            max_activities=self.max_activities,
            max_resources=self.max_resources,
            capacity_scale=self._capacity_scales[self._active_index],
            out=buffer,
        )
