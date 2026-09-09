"""Factories for RCMPSP environments used by SB3 training and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from src.environments.multi_instance import MultiInstanceRCMPSPEnv
from src.environments.sb3_env import make_sb3_env
from src.training.callbacks import TERMINAL_METRICS


def monitored_env(env) -> Monitor:
    """Expose episode returns and RCMPSP terminal metrics to SB3 logging."""
    return Monitor(env, info_keywords=TERMINAL_METRICS)


def make_single_env(instance: str | Path) -> Monitor:
    """Create a monitored, flattened environment for one instance."""
    return monitored_env(make_sb3_env(instance))


def make_multi_env(
    instance_paths: list[str],
    *,
    max_activities: int | None = None,
    max_resources: int | None = None,
    instance_indices: list[int] | None = None,
    catalog_size: int | None = None,
) -> Monitor:
    """Create a monitored multi-instance environment with optional padding."""
    return monitored_env(
        MultiInstanceRCMPSPEnv(
            instance_paths,
            max_activities=max_activities,
            max_resources=max_resources,
            instance_indices=instance_indices,
            catalog_size=catalog_size,
        )
    )


def make_vector_env(
    env_fns: list[Callable[[], object]],
    *,
    backend: str = "auto",
    start_method: str = "spawn",
) -> VecEnv:
    """Build either an in-process or subprocess SB3 vector environment."""
    if backend not in {"auto", "dummy", "subproc"}:
        raise ValueError("backend must be one of: auto, dummy, subproc")
    if backend == "auto":
        backend = "subproc" if len(env_fns) > 1 else "dummy"
    if backend == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=start_method)
