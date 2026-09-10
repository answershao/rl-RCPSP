"""PPO configuration and evaluation helpers for RCPSP experiments."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

from src.core.rcpsp import Instance
from src.data.adapter import load_core_instance
from src.envs.observation import (
    MAX_SUCCESSORS,
    ObservationLayout,
    StaticGraphCache,
    build_static_graph_cache,
)
from src.training.features import GINActorCriticHeads, SharedDirectedGINExtractor


class GINActorCriticPolicy(ActorCriticPolicy):
    """PPO policy with a shared GIN trunk and independent task-specific heads."""

    def _build_mlp_extractor(self) -> None:
        extractor = self.features_extractor
        if not isinstance(extractor, SharedDirectedGINExtractor):
            raise TypeError("GINActorCriticPolicy requires SharedDirectedGINExtractor")
        self.mlp_extractor = GINActorCriticHeads(
            max_activities=extractor.max_activities,
            embedding_dim=extractor.embedding_dim,
            global_dim=extractor.global_dim,
            mixed_precision=extractor.mixed_precision,
        )

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        # The actor head already emits one masked logit per discrete action and
        # the critic head already emits a scalar value.
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        if self.ortho_init:
            self.features_extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.mlp_extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))
            self.mlp_extractor.actor[-1].apply(partial(self.init_weights, gain=0.01))
            self.mlp_extractor.critic[-1].apply(partial(self.init_weights, gain=1.0))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


MAX_SAMPLE_TRAJECTORIES = 32


def _validate_sample_budgets(sample_budgets: Sequence[int]) -> tuple[int, ...]:
    budgets = tuple(int(budget) for budget in sample_budgets)
    if not budgets:
        raise ValueError("sample_budgets must not be empty")
    if len(set(budgets)) != len(budgets):
        raise ValueError("sample_budgets must not contain duplicates")
    if any(budget < 1 or budget > MAX_SAMPLE_TRAJECTORIES for budget in budgets):
        raise ValueError(
            f"sample_budgets must be between 1 and {MAX_SAMPLE_TRAJECTORIES}"
        )
    return budgets


def _cache_from_extractor(extractor: SharedDirectedGINExtractor) -> StaticGraphCache:
    """Snapshot the extractor cache before temporarily evaluating another split."""
    arrays = {
        name: getattr(extractor, name).detach().cpu().numpy().copy()
        for name in (
            "static_durations",
            "static_resource_demands",
            "static_successor_indices",
            "static_successor_counts",
            "static_predecessor_counts",
            "static_downstream_durations",
            "static_activity_mask",
            "static_slack_ratios",
            "static_on_critical_path",
        )
    }
    return StaticGraphCache(
        instance_names=tuple(extractor.instance_names),
        durations=arrays["static_durations"],
        resource_demands=arrays["static_resource_demands"],
        successor_indices=arrays["static_successor_indices"],
        successor_counts=arrays["static_successor_counts"],
        predecessor_counts=arrays["static_predecessor_counts"],
        downstream_durations=arrays["static_downstream_durations"],
        activity_mask=arrays["static_activity_mask"],
        slack_ratios=arrays["static_slack_ratios"],
        on_critical_path=arrays["static_on_critical_path"],
    )


def create_ppo(
    env,
    *,
    instances: list[Instance],
    seed: int,
    device: str = "auto",
    n_steps: int = 256,
    batch_size: int = 1024,
    n_epochs: int = 3,
    # See scripts/train_ppo.py: gamma=1 makes the return exactly -makespan/scale.
    gamma: float = 1.0,
    gae_lambda: float = 0.98,
    learning_rate: float = 2e-4,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    target_kl: float | None = None,
    gin_layers: int = 2,
    mixed_precision: str = "none",
    torch_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    tensorboard_log: str | None = None,
    static_cache: StaticGraphCache | None = None,
) -> PPO:
    """Create PPO with a shared directed GIN and masked discrete actor.

    The actor selects one eligible activity and the environment inserts it at
    its earliest precedence- and resource-feasible start time.
    """
    if n_steps < 1 or batch_size < 1 or n_epochs < 1 or gin_layers < 1:
        raise ValueError("n_steps, batch_size, n_epochs, and gin_layers must be positive")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be between 0 and 1")
    if learning_rate <= 0 or ent_coef < 0 or vf_coef < 0:
        raise ValueError("learning_rate must be positive; ent_coef and vf_coef non-negative")
    if target_kl is not None and target_kl <= 0:
        raise ValueError("target_kl must be positive when configured")
    if not instances:
        raise ValueError("instances must not be empty")
    if mixed_precision not in {"none", "bf16", "fp16"}:
        raise ValueError("mixed_precision must be one of: none, bf16, fp16")
    action_count = int(env.action_space.n)
    base_env = env.envs[0] if hasattr(env, "envs") else env
    max_activities = action_count
    observation_dim = int(env.observation_space.shape[0])
    one_resource_size = ObservationLayout(max_activities, 1).size
    per_resource_size = ObservationLayout(max_activities, 2).size - one_resource_size
    resource_payload = observation_dim - one_resource_size
    if resource_payload <= 0 or resource_payload % per_resource_size:
        raise ValueError(
            "environment observation has incompatible RCPSP layout: "
            f"got {observation_dim} for {max_activities} activities"
        )
    max_resources = resource_payload // per_resource_size + 1
    if max_resources < max(instance.resource_count for instance in instances):
        raise ValueError("environment observation cannot represent all instance resources")
    max_successors = getattr(base_env, "max_successors", MAX_SUCCESSORS)
    if static_cache is None:
        static_cache = build_static_graph_cache(
            instances,
            max_activities=max_activities,
            max_resources=max_resources,
            max_successors=max_successors,
        )
    expected_instance_names = tuple(instance.name for instance in instances)
    if static_cache.instance_names != expected_instance_names:
        raise ValueError(
            "static cache ordering does not match the PPO instance catalog"
        )
    model = PPO(
        GINActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        target_kl=target_kl,
        policy_kwargs={
            "features_extractor_class": SharedDirectedGINExtractor,
            "features_extractor_kwargs": {
                "max_activities": max_activities,
                "max_resources": max_resources,
                "static_cache": static_cache,
                "max_successors": max_successors,
                "gin_layers": gin_layers,
                "embedding_dim": 32,
                "hidden_dim": 64,
                "global_dim": 16,
                "mixed_precision": mixed_precision,
            },
        },
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )
    if torch_compile:
        # Compile only the compute-heavy modules. Keeping the SB3 policy itself
        # unwrapped preserves its save/load and callback interfaces.
        try:
            model.policy.features_extractor.compile(mode=compile_mode, dynamic=True)
            model.policy.mlp_extractor.compile(mode=compile_mode, dynamic=True)
        except Exception as exc:  # noqa: BLE001 - compile is an optimisation only
            # An inductor backend failure must never take the run down; the
            # caller's probe normally catches this earlier (see
            # scripts/train_ppo.py::configure_torch_runtime).
            print(f"torch.compile failed ({type(exc).__name__}: {exc}); "
                  "continuing with eager execution")
    return model


def evaluate_paths(
    model: PPO,
    paths: list[str],
    seed: int,
    reference_env,
    *,
    batch_size: int = 32,
    evaluation_cache: StaticGraphCache | None = None,
    restore_cache: StaticGraphCache | None = None,
    loader=None,
    name_fn=None,
) -> list[tuple[str, float]]:
    """Evaluate paths in inference batches while keeping environments independent.

    The static graph cache is keyed by *instance name* (see
    :func:`src.envs.observation.build_static_graph_cache`).  ``name_fn`` maps an
    evaluation path to the cache key that was used when the cache was built, so
    the two must agree exactly:

    * single-process / unique-stem usage may keep the default ``Path.stem``;
    * protocol callers must pass ``name_fn=instance_id`` and the matching
      protocol loader, or this check fails loudly instead of silently reusing a
      cache that maps paths to the wrong activities.

    ``evaluation_cache``/``restore_cache`` switch the extractor's static cache
    for the duration of the evaluation and restore the previous one afterwards.
    """
    from src.envs.multi_instance import MultiInstanceRCPSPEnv

    loader = loader or load_core_instance
    name_fn = name_fn or (lambda path: path.stem)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not paths:
        return []
    extractor = model.policy.features_extractor
    if not isinstance(extractor, SharedDirectedGINExtractor):
        raise TypeError("model does not use the RCPSP static graph cache")
    original_cache = _cache_from_extractor(extractor)
    if evaluation_cache is None:
        evaluation_instances = [loader(Path(path)) for path in paths]
        evaluation_cache = build_static_graph_cache(
            evaluation_instances,
            max_activities=reference_env.max_activities,
            max_resources=reference_env.max_resources,
            max_successors=extractor.max_successors,
        )
    # name_fn must reproduce the cache keys exactly; a mismatch means the cache
    # was built from different paths or a different name convention (stem vs
    # protocol instance_id), so fail before any observation is generated.
    expected_names = tuple(name_fn(Path(path)) for path in paths)
    if evaluation_cache.instance_names != expected_names:
        raise ValueError(
            "evaluation cache instance names "
            f"{evaluation_cache.instance_names!r} do not match evaluation paths "
            f"under name_fn {expected_names!r}; pass loader/name_fn consistent "
            "with the cache (protocol callers: instances.instance_id / loader_for)"
        )
    extractor.set_static_cache(evaluation_cache)
    try:
        makespans = np.zeros(len(paths), dtype=np.float64)
        for batch_start in range(0, len(paths), batch_size):
            batch_paths = paths[batch_start : batch_start + batch_size]
            envs = [
                MultiInstanceRCPSPEnv(
                    [path],
                    max_activities=reference_env.max_activities,
                    max_resources=reference_env.max_resources,
                    instance_indices=[batch_start + local_index],
                    catalog_size=extractor.instance_count,
                    loader=loader,
                )
                for local_index, path in enumerate(batch_paths)
            ]
            try:
                observations = [
                    env.reset(seed=seed + batch_start + local_index)[0]
                    for local_index, env in enumerate(envs)
                ]
                active = np.ones(len(envs), dtype=np.bool_)
                while active.any():
                    active_indices = np.flatnonzero(active)
                    observation_batch = np.stack([observations[index] for index in active_indices])
                    actions, _ = model.predict(observation_batch, deterministic=True)
                    for local_index, action in zip(active_indices, np.asarray(actions).reshape(-1)):
                        observation, _, terminated, truncated, info = envs[local_index].step(
                            int(action)
                        )
                        observations[local_index] = observation
                        if terminated or truncated:
                            makespans[batch_start + local_index] = float(info["makespan"])
                            active[local_index] = False
            finally:
                for env in envs:
                    env.close()
            completed = min(batch_start + batch_size, len(paths))
            print(f"evaluation progress: {completed}/{len(paths)}")
    finally:
        extractor.set_static_cache(restore_cache or original_cache)
    return [(name_fn(Path(path)), makespans[index]) for index, path in enumerate(paths)]


def evaluate_paths_sampled(
    model: PPO,
    paths: list[str],
    seed: int,
    reference_env,
    *,
    sample_budgets: Sequence[int] = (8, 16, 32),
    batch_size: int = 32,
    evaluation_cache: StaticGraphCache | None = None,
    restore_cache: StaticGraphCache | None = None,
    path_offset: int = 0,
    loader=None,
    name_fn=None,
) -> dict[int, list[tuple[str, float]]]:
    """Evaluate stochastic policy trajectories and keep the best makespan.

    Each instance always generates the same prefix of up to 32 trajectories.
    Results for smaller budgets are therefore prefixes of the 32-trajectory
    run, rather than independent random experiments.
    """
    from src.envs.multi_instance import MultiInstanceRCPSPEnv

    loader = loader or load_core_instance
    name_fn = name_fn or (lambda path: path.stem)
    budgets = _validate_sample_budgets(sample_budgets)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if path_offset < 0:
        raise ValueError("path_offset must be non-negative")
    if not paths:
        return {budget: [] for budget in budgets}

    extractor = model.policy.features_extractor
    if not isinstance(extractor, SharedDirectedGINExtractor):
        raise TypeError("model does not use the RCPSP static graph cache")
    original_cache = _cache_from_extractor(extractor)
    max_samples = max(budgets)
    if evaluation_cache is None:
        evaluation_instances = [loader(Path(path)) for path in paths]
        evaluation_cache = build_static_graph_cache(
            evaluation_instances,
            max_activities=reference_env.max_activities,
            max_resources=reference_env.max_resources,
            max_successors=extractor.max_successors,
        )
    expected_names = tuple(name_fn(Path(path)) for path in paths)
    if evaluation_cache.instance_names != expected_names:
        raise ValueError("evaluation cache does not match evaluation paths")

    # SB3 samples actions from torch's global RNG. Isolating and seeding it
    # makes repeated search runs reproducible without changing caller state.
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    extractor.set_static_cache(evaluation_cache)
    try:
        makespans = np.full(
            (len(paths), max_samples), np.inf, dtype=np.float64
        )
        for batch_start in range(0, len(paths), batch_size):
            batch_paths = paths[batch_start : batch_start + batch_size]
            envs = []
            trajectory_paths: list[int] = []
            trajectory_numbers: list[int] = []
            for local_index, path in enumerate(batch_paths):
                for trajectory_number in range(max_samples):
                    envs.append(
                        MultiInstanceRCPSPEnv(
                            [path],
                            max_activities=reference_env.max_activities,
                            max_resources=reference_env.max_resources,
                            instance_indices=[batch_start + local_index],
                            catalog_size=extractor.instance_count,
                            loader=loader,
                        )
                    )
                    trajectory_paths.append(local_index)
                    trajectory_numbers.append(trajectory_number)
            try:
                observations = [
                    env.reset(
                        seed=seed
                        + (path_offset + batch_start + path_index) * max_samples
                        + trajectory_number
                    )[0]
                    for env, path_index, trajectory_number in zip(
                        envs, trajectory_paths, trajectory_numbers
                    )
                ]
                active = np.ones(len(envs), dtype=np.bool_)
                while active.any():
                    active_indices = np.flatnonzero(active)
                    observation_batch = np.stack(
                        [observations[index] for index in active_indices]
                    )
                    actions, _ = model.predict(
                        observation_batch, deterministic=False
                    )
                    for local_env_index, action in zip(
                        active_indices, np.asarray(actions).reshape(-1)
                    ):
                        observation, _, terminated, truncated, info = envs[
                            local_env_index
                        ].step(int(action))
                        observations[local_env_index] = observation
                        if terminated or truncated:
                            makespans[
                                trajectory_paths[local_env_index],
                                trajectory_numbers[local_env_index],
                            ] = float(info["makespan"])
                            active[local_env_index] = False
            finally:
                for env in envs:
                    env.close()
            completed = min(batch_start + batch_size, len(paths))
            print(
                f"sampled evaluation progress: {completed}/{len(paths)} "
                f"({max_samples} trajectories)"
            )

        result: dict[int, list[tuple[str, float]]] = {}
        for budget in budgets:
            best = np.min(makespans[:, :budget], axis=1)
            result[budget] = [
                (name_fn(Path(path)), float(best[index]))
                for index, path in enumerate(paths)
            ]
        return result
    finally:
        extractor.set_static_cache(restore_cache or original_cache)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
