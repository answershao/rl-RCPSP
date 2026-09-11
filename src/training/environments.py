"""Factories for RCPSP environments used by SB3 training and evaluation."""

from __future__ import annotations

from collections.abc import Callable

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from src.envs.multi_instance import MultiInstanceRCPSPEnv
from src.training.callbacks import TERMINAL_METRICS


def monitored_env(env) -> Monitor:
    """Expose episode returns and RCPSP terminal metrics to SB3 logging."""
    return Monitor(env, info_keywords=TERMINAL_METRICS)


def make_multi_env(
    instance_paths: list[str],
    *,
    max_activities: int | None = None,
    max_resources: int | None = None,
    max_horizon: int | None = None,
    instance_indices: list[int] | None = None,
    catalog_size: int | None = None,
    reward_shaping_coef: float = 0.0,
    loader: Callable | None = None,
) -> Monitor:
    """Create a monitored multi-instance environment with optional padding."""
    kwargs = dict(
        instances=instance_paths,
        max_activities=max_activities,
        max_resources=max_resources,
        max_horizon=max_horizon,
        instance_indices=instance_indices,
        catalog_size=catalog_size,
        reward_shaping_coef=reward_shaping_coef,
    )
    if loader is not None:
        kwargs["loader"] = loader
    return monitored_env(MultiInstanceRCPSPEnv(**kwargs))


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
