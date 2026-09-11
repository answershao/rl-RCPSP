#!/usr/bin/env python3
"""Measure PPO training throughput (steps/s) on the host that will train.

A PPO iteration splits into the rollout (environment sampling plus one forward
pass per decision) and the update (``n_epochs`` of forward+backward over the
collected batch).  The update dominates, and its cost scales with the *padded*
graph size and the minibatch size, so those two knobs plus the torch thread
count matter far more than the environment itself.

This script sweeps them on the real machine and prints rollout / update / total
throughput per combination, so ``train_cpu.sh`` and ``train_a800.sh`` can be
tuned from measurements instead of guesses::

    python -m scripts.bench_ppo --caps 122 --threads 8 16 20 \
        --batch-sizes 512 1024 4096

The main protocol uses one 122-node cap for generated training data and held-out
PSPLIB j30-j120 evaluation data.

Match the production configuration before trusting the numbers.  ``train_cpu.sh``
runs with ``--torch-compile --compile-mode default`` and ``--vec-env subproc
--start-method spawn``.  Pass the same flags here: without ``--torch-compile``
the update is measured in eager mode, and with the default ``--vec-env dummy``
the rollout runs in a single process, so both phases come out pessimistic and
the best thread count can land on the wrong value::

    python -m scripts.bench_ppo --caps 122 --threads 8 16 20 24 32 \
        --batch-sizes 1024 --torch-compile --compile-mode default \
        --vec-env subproc --start-method spawn --repeats 3
"""
from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_baselines3.common.callbacks import BaseCallback

from src.data.instances import loader_for, read_protocol
from src.envs.multi_instance import MultiInstanceRCPSPEnv
from src.envs.observation import build_static_graph_cache, observation_size
from src.training.environments import make_vector_env
from src.training.ppo import create_ppo


class _NoopCallback(BaseCallback):
    """``collect_rollouts`` requires an initialised callback."""

    def _on_step(self) -> bool:
        return True


class _FixedInstanceLoader:
    """Picklable replacement for the ``loader=lambda _path, item=...: item`` closure.

    ``dummy`` steps every environment in-process, so a closure is harmless there.
    ``subproc`` pickles each ``env_fn`` to a spawned worker instead, and a lambda
    cannot be pickled, so the instance is carried in an object.
    """

    def __init__(self, instance) -> None:
        self.instance = instance

    def __call__(self, _path):
        return self.instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits", type=Path, default=Path("splits.json"))
    parser.add_argument("--instances", type=int, default=64,
                        help="training instances to load into the static catalog")
    parser.add_argument("--caps", type=int, nargs="+", default=[122],
                        help="padded activity caps to compare")
    parser.add_argument("--threads", type=int, nargs="+", default=[8, 16, 20],
                        help="torch intra-op threads for the training process")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[512, 1024, 4096])
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=384)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--gin-layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=1,
                        help="time each combination N times and keep the best")
    parser.add_argument("--vec-env", choices=("dummy", "subproc"), default="dummy",
                        help="mirror train_cpu.sh's --vec-env; subproc needs picklable env fns")
    parser.add_argument("--start-method", default="spawn",
                        help="start method for --vec-env subproc")
    parser.add_argument("--torch-compile", action="store_true",
                        help="compile features_extractor and mlp_extractor, as train_cpu.sh does")
    parser.add_argument("--compile-mode", default="default",
                        help="torch.compile mode; train_cpu.sh uses 'default' on CPU")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def measure(
    args: argparse.Namespace,
    instances,
    cap: int,
    max_resources: int,
    threads: int,
    batch_size: int,
) -> dict:
    """Time one rollout + update pair for a single configuration."""
    torch.set_num_threads(threads)
    cache = build_static_graph_cache(
        instances, max_activities=cap, max_resources=max_resources
    )
    max_horizon = max(
        sum(activity.duration for activity in item.activities.values())
        for item in instances
    )
    env = make_vector_env(
        [
            partial(
                MultiInstanceRCPSPEnv,
                [instance.name],
                max_activities=cap,
                max_resources=max_resources,
                max_horizon=max_horizon,
                instance_indices=[index],
                catalog_size=len(instances),
                loader=_FixedInstanceLoader(instance),
            )
            for index, instance in enumerate(instances[: args.n_envs])
        ],
        backend=args.vec_env,
        start_method=args.start_method,
    )
    try:
        model = create_ppo(
            env,
            instances=list(instances),
            seed=1,
            device=args.device,
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gin_layers=args.gin_layers,
            static_cache=cache,
            torch_compile=args.torch_compile,
            compile_mode=args.compile_mode,
        )
        model.verbose = 0
        callback = _NoopCallback()
        model._setup_learn(
            total_timesteps=10_000_000, callback=callback, progress_bar=False
        )
        # One untimed iteration absorbs lazy allocation, kernel warm-up, and the
        # one-off torch.compile graph build of *both* phases.  The update half is
        # needed because the compiled update is only realised on its first call;
        # timing it away here keeps the first repeat from being discarded.
        model.collect_rollouts(env, callback, model.rollout_buffer, model.n_steps)
        model.train()

        rollout_seconds = update_seconds = float("inf")
        for _ in range(args.repeats):
            started = time.perf_counter()
            model.collect_rollouts(env, callback, model.rollout_buffer, model.n_steps)
            rollout_seconds = min(rollout_seconds, time.perf_counter() - started)
            started = time.perf_counter()
            model.train()
            update_seconds = min(update_seconds, time.perf_counter() - started)
    finally:
        env.close()

    steps = args.n_envs * args.n_steps
    return {
        "cap": cap,
        "threads": threads,
        "batch_size": batch_size,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "gin_layers": args.gin_layers,
        "vec_env": args.vec_env,
        "torch_compile": int(args.torch_compile),
        "compile_mode": args.compile_mode if args.torch_compile else "",
        "obs_dim": observation_size(
            cap,
            max_resources,
            max(
                sum(activity.duration for activity in item.activities.values())
                for item in instances
            ),
        ),
        "rollout_seconds": round(rollout_seconds, 3),
        "update_seconds": round(update_seconds, 3),
        "rollout_fps": round(steps / rollout_seconds, 1),
        "update_fps": round(steps / update_seconds, 1),
        "total_fps": round(steps / (rollout_seconds + update_seconds), 1),
    }


def main() -> None:
    args = parse_args()
    if args.instances < 1 or args.repeats < 1:
        raise ValueError("--instances and --repeats must be positive")
    if args.n_envs > args.instances:
        raise ValueError("--n-envs cannot exceed --instances")
    if min(args.caps) < 1 or min(args.threads) < 1 or min(args.batch_sizes) < 1:
        raise ValueError("--caps, --threads, and --batch-sizes must be positive")
    rollout_size = args.n_envs * args.n_steps
    oversized = [size for size in args.batch_sizes if size > rollout_size]
    if oversized:
        raise ValueError(
            f"--batch-sizes {oversized} exceed the rollout size {rollout_size}"
        )

    protocol = read_protocol(args.splits)
    loader = loader_for(args.data_root)
    rels = protocol["train"][: args.instances]
    if not rels:
        raise ValueError("the training split is empty")
    instances = [loader(rel) for rel in rels]
    max_resources = max(instance.resource_count for instance in instances)
    print(
        f"host benchmark: instances={len(instances)} "
        f"activities={max(len(i.activities) for i in instances)} "
        f"resources={max_resources} device={args.device} "
        f"rollout={rollout_size} epochs={args.n_epochs} "
        f"vec_env={args.vec_env} compile={args.torch_compile}"
        f"{'/' + args.compile_mode if args.torch_compile else ''}"
    )

    rows = []
    for cap in args.caps:
        for threads in args.threads:
            for batch_size in args.batch_sizes:
                row = measure(args, instances, cap, max_resources, threads, batch_size)
                rows.append(row)
                print(
                    f"cap={cap:4d} threads={threads:3d} batch={batch_size:5d} "
                    f"rollout={row['rollout_seconds']:7.2f}s ({row['rollout_fps']:8.1f} fps) "
                    f"update={row['update_seconds']:7.2f}s ({row['update_fps']:8.1f} fps) "
                    f"total={row['total_fps']:8.1f} fps",
                    flush=True,
                )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.output_csv}")

    best = max(rows, key=lambda row: row["total_fps"])
    print(
        "best: "
        f"cap={best['cap']} threads={best['threads']} batch={best['batch_size']} "
        f"-> {best['total_fps']} steps/s"
    )


if __name__ == "__main__":
    main()
