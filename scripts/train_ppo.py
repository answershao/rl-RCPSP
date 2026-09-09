#!/usr/bin/env python3
"""Train the graph-aware PPO policy on the configured RCMPSP training split."""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from src.core.rcmpsp import parse_rcmp
from src.environments.multi_instance import (
    DEFAULT_INSTANCES_ROOT,
    make_overfit_splits,
    make_splits,
    partition_instance_catalog,
    write_split_manifest,
    write_splits,
)
from src.environments.observation import build_static_graph_cache, observation_size
from src.training.callbacks import RCMPSPMetricsCallback
from src.training.environments import make_multi_env, make_vector_env
from src.training.ppo import create_ppo, evaluate_paths
from scripts.compare_ppo_results import compare_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=6_400_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.0,
        help="stop PPO epochs early above this approximate KL; 0 disables it",
    )
    parser.add_argument(
        "--gin-layers",
        type=int,
        default=2,
        help="Number of directed GIN message-passing layers.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mixed-precision", choices=("none", "bf16", "fp16"), default="none",
        help="Autocast GIN and policy heads on CUDA; bf16 is recommended for A800.",
    )
    parser.add_argument(
        "--torch-compile", action="store_true",
        help="Compile the GIN and policy heads (initial iterations include compilation time).",
    )
    parser.add_argument(
        "--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--cuda-matmul-precision", choices=("highest", "high", "medium"), default="high",
        help="PyTorch float32 matrix multiplication precision on CUDA; high enables TF32.",
    )
    parser.add_argument("--vec-env", choices=("auto", "dummy", "subproc"), default="auto")
    parser.add_argument(
        "--start-method", choices=("spawn", "forkserver", "fork"), default="spawn",
        help="Multiprocessing start method used by the subproc vector environment.",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop after N validation evaluations without improvement; 0 disables it",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=10,
        help="evaluate the validation split every N rollouts",
    )
    parser.add_argument(
        "--validation-min-delta",
        "--makespan-min-delta",
        dest="validation_min_delta",
        type=float,
        default=0.0,
        help="minimum mean FIFO-relative-gap improvement required to reset patience",
    )
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=DEFAULT_INSTANCES_ROOT,
        help="directory containing the .rcmp training instances",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/experiments/ppo/ppo_gnn"),
    )
    parser.add_argument(
        "--splits", type=Path,
        default=Path("outputs/manifests/ppo_gnn/splits.json"),
    )
    parser.add_argument(
        "--overfit-instances",
        type=int,
        default=0,
        help=(
            "sample one training replicate from each of N MPLIB2 parameter groups, "
            "and validate on those same instances; 0 uses the standard split"
        ),
    )
    parser.add_argument(
        "--eval-max-instances",
        type=int,
        default=0,
        help="evaluate at most N instances per split; 0 evaluates all",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="number of independent instances advanced per policy inference batch",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and evaluate output-dir/final_model.zip.",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path("outputs/baselines_mplib2_10_50_5/makespan_summary.csv"),
        help="shared FIFO, Shortest, Random, and CP-SAT results produced by baselines.py",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="reference makespan_summary.csv to compare after evaluation",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA {requested!r} unavailable; falling back to CPU")
        return "cpu"
    if requested in {"cpu", "cuda"} or requested.startswith("cuda:"):
        return requested
    raise ValueError("--device must be one of: auto, cpu, cuda, cuda:N")


def configure_torch_runtime(args: argparse.Namespace) -> None:
    """Configure CPU threads and CUDA math without changing PPO semantics."""
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    if args.torch_compile and shutil.which("g++") is None:
        print("g++ is unavailable; disabling torch.compile and using eager execution")
        args.torch_compile = False
    if not args.device.startswith("cuda"):
        if args.mixed_precision != "none":
            print("mixed precision requested without CUDA; disabling it")
            args.mixed_precision = "none"
        if args.torch_compile:
            print("warning: torch.compile on CPU can reduce throughput for this small policy")
        return
    torch.set_float32_matmul_precision(args.cuda_matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = args.cuda_matmul_precision != "highest"
    if args.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support bf16")


BASELINE_METHODS = ("FIFO", "Shortest", "Random", "CP-SAT")


def load_baseline_results(
    path: Path, expected_instances: list[str]
) -> dict[str, dict[str, int]]:
    """Load and validate shared baseline and exact-solver results."""
    if not path.is_file():
        raise FileNotFoundError(
            f"baseline results not found: {path}; run scripts.baselines first"
        )
    with path.open(newline="", encoding="ascii") as result_file:
        reader = csv.DictReader(result_file)
        required = {"instance", *BASELINE_METHODS}
        if missing_fields := required - set(reader.fieldnames or ()):
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_fields)}"
            )
        results = {}
        for raw in reader:
            name = raw["instance"]
            if name in results:
                raise ValueError(f"{path} contains duplicate instance {name!r}")
            try:
                results[name] = {
                    method: int(raw[method]) for method in BASELINE_METHODS
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} contains invalid values for {name!r}") from exc

    missing_instances = sorted(set(expected_instances) - results.keys())
    if missing_instances:
        raise ValueError(
            f"{path} does not cover {len(missing_instances)} requested instances; "
            f"first missing instance: {missing_instances[0]}"
        )
    return results


def evaluate_and_save_results(
    model: PPO, splits: dict[str, list[str]], seed: int, reference_env,
    output_dir: Path, baseline_results: dict[str, dict[str, int]],
    max_eval_instances: int = 0,
    eval_batch_size: int = 32,
) -> Path:
    """Write one makespan matrix row for each evaluated RCMPSP instance."""
    if max_eval_instances:
        splits = {
            split: paths[:max_eval_instances] for split, paths in splits.items()
        }
    rows = []
    rows_by_split: dict[str, list[dict[str, int | str]]] = {
        split: [] for split in splits
    }
    for split, paths in splits.items():
        ppo_by_name = dict(
            evaluate_paths(
                model,
                paths,
                seed,
                reference_env,
                batch_size=eval_batch_size,
            )
        )
        for path in paths:
            instance_name = Path(path).name
            row = {
                "instance": instance_name,
                "PPO": int(ppo_by_name[instance_name]),
                **baseline_results[instance_name],
            }
            rows.append(row)
            rows_by_split[split].append(row)
            print(
                f"{split} {row['instance']}: PPO={row['PPO']} FIFO={row['FIFO']} "
                f"Shortest={row['Shortest']} Random={row['Random']} "
                f"CP-SAT={row['CP-SAT']} (cached)"
            )

    result_path = output_dir / "makespan_summary.csv"
    if not rows:
        print("no instances configured; skipping makespan summary")
        return result_path
    with result_path.open("w", newline="", encoding="ascii") as result_file:
        methods = ["PPO", *BASELINE_METHODS]
        writer = csv.DictWriter(result_file, fieldnames=["instance", *methods])
        writer.writeheader()
        writer.writerows(rows)
    print(f"makespan summary: {result_path}")
    for split, split_rows in rows_by_split.items():
        if split_rows:
            summary = (
                f"{split}: PPO_mean={np.mean([row['PPO'] for row in split_rows]):.2f} "
                f"FIFO_mean={np.mean([row['FIFO'] for row in split_rows]):.2f} "
                f"Shortest_mean={np.mean([row['Shortest'] for row in split_rows]):.2f} "
                f"Random_mean={np.mean([row['Random'] for row in split_rows]):.2f} "
                f"CP-SAT_mean={np.mean([row['CP-SAT'] for row in split_rows]):.2f}"
            )
            print(summary)
    return result_path


def evaluate_fifo_relative_gap(
    model: PPO,
    paths: list[str],
    seed: int,
    reference_env,
    baseline_results: dict[str, dict[str, int]],
    *,
    batch_size: int,
    evaluation_cache,
    restore_cache,
) -> float:
    """Evaluate deterministic validation schedules relative to FIFO per instance."""
    results = evaluate_paths(
        model,
        paths,
        seed,
        reference_env,
        batch_size=batch_size,
        evaluation_cache=evaluation_cache,
        restore_cache=restore_cache,
    )
    gaps = []
    for instance_name, makespan in results:
        fifo = baseline_results[instance_name]["FIFO"]
        if fifo <= 0:
            raise ValueError(f"FIFO makespan must be positive for {instance_name}")
        gaps.append((makespan - fifo) / fifo)
    gap = float(np.mean(gaps))
    print(f"validation: FIFO-relative gap={gap:+.6f} over {len(gaps)} instances")
    return gap


def main() -> None:
    args = parse_args()
    if (
        args.total_timesteps < 1
        or args.n_envs < 1
        or args.n_steps < 1
        or args.batch_size < 1
        or args.n_epochs < 1
        or not 0.0 <= args.gamma <= 1.0
        or not 0.0 <= args.gae_lambda <= 1.0
        or args.learning_rate <= 0
        or args.ent_coef < 0
        or args.vf_coef < 0
        or args.target_kl < 0
        or args.gin_layers < 1
        or args.torch_threads < 1
        or args.torch_interop_threads < 1
        or args.early_stop_patience < 0
        or args.validation_interval < 1
        or args.validation_min_delta < 0
    ):
        raise ValueError(
            "timesteps, n-envs, n-steps, and batch-size must be positive; "
            "n-epochs, gin-layers, and torch thread counts must be positive; "
            "gamma and gae-lambda must be between 0 and 1; learning-rate must "
            "be positive; ent-coef, vf-coef, and target-kl must be non-negative"
        )
    if (
        args.eval_max_instances < 0
        or args.eval_batch_size < 1
        or args.overfit_instances < 0
    ):
        raise ValueError(
            "--eval-max-instances and --overfit-instances must be non-negative; "
            "--eval-batch-size must be positive"
        )
    rollout_size = args.n_envs * args.n_steps
    if rollout_size % args.batch_size:
        raise ValueError("batch-size must divide n-envs * n-steps")
    args.device = resolve_device(args.device)
    configure_torch_runtime(args)
    set_random_seed(args.seed)

    splits = make_splits(args.instances_root)
    if args.overfit_instances:
        splits = make_overfit_splits(
            splits, args.overfit_instances, seed=args.seed
        )
        write_split_manifest(splits, args.splits)
        print(
            f"overfit diagnostic: {len(splits['train'])} grouped training instances; "
            "validation reuses the training instances"
        )
    else:
        write_splits(args.splits, root=args.instances_root)
    train_paths = splits["train"]
    catalog_paths = list(dict.fromkeys(path for paths in splits.values() for path in paths))
    baseline_results = load_baseline_results(
        args.baseline_results, [Path(path).name for path in catalog_paths]
    )
    if args.n_envs > len(train_paths):
        raise ValueError(
            f"--n-envs ({args.n_envs}) cannot exceed the number of training "
            f"instances ({len(train_paths)})"
        )
    instances_by_path = {path: parse_rcmp(path) for path in catalog_paths}
    train_instances = [instances_by_path[path] for path in train_paths]
    validation_paths = splits["validation"]
    validation_instances = [instances_by_path[path] for path in validation_paths]
    if not validation_instances:
        raise ValueError("validation split must not be empty")
    all_instances = list(instances_by_path.values())
    max_activities = max(len(instance.activities) for instance in all_instances)
    max_resources = max(instance.resource_count for instance in all_instances)
    reference_env = SimpleNamespace(
        max_activities=max_activities,
        max_resources=max_resources,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = (
        f"device={args.device}; envs={args.n_envs}; vec_env={args.vec_env}; "
        f"rollout={rollout_size}; batch={args.batch_size}; epochs={args.n_epochs}; "
        f"lr={args.learning_rate}; ent_coef={args.ent_coef}; vf_coef={args.vf_coef}; "
        f"target_kl={args.target_kl or 'disabled'}; gamma={args.gamma}; "
        f"gae_lambda={args.gae_lambda}; amp={args.mixed_precision}; "
        f"compile={args.torch_compile}; objective=makespan_only; "
        f"obs_dim={observation_size(max_activities, max_resources)}"
    )
    if args.device.startswith("cuda"):
        runtime += f"; gpu={torch.cuda.get_device_name(torch.device(args.device))}"
    print(runtime)
    if args.evaluate_only:
        model_path = args.output_dir / "final_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"PPO model not found: {model_path}")
        model = PPO.load(str(model_path), device=args.device)
        result_splits = {"train": train_paths} if args.overfit_instances else splits
        result_path = evaluate_and_save_results(
            model, result_splits, args.seed, reference_env, args.output_dir,
            baseline_results,
            max_eval_instances=args.eval_max_instances,
            eval_batch_size=args.eval_batch_size,
        )
        if args.compare_with is not None:
            compare_results(
                args.compare_with,
                result_path,
                args.output_dir / "comparison.csv",
            )
        return

    worker_catalogs = partition_instance_catalog(train_paths, args.n_envs)
    env_fns = [
        partial(
            make_multi_env,
            worker_paths,
            max_activities=max_activities,
            max_resources=max_resources,
            instance_indices=worker_indices,
            catalog_size=len(train_paths),
        )
        for worker_paths, worker_indices in worker_catalogs
    ]
    env = make_vector_env(
        env_fns,
        backend=args.vec_env,
        start_method=args.start_method,
    )
    training_cache = build_static_graph_cache(
        train_instances,
        max_activities=max_activities,
        max_resources=max_resources,
    )
    validation_cache = build_static_graph_cache(
        validation_instances,
        max_activities=max_activities,
        max_resources=max_resources,
    )
    model = create_ppo(
        env,
        instances=train_instances,
        seed=args.seed,
        device=args.device,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl or None,
        gin_layers=args.gin_layers,
        mixed_precision=args.mixed_precision,
        torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
        tensorboard_log=str(args.output_dir / "tensorboard"),
        static_cache=training_cache,
    )
    validation_evaluator = partial(
        evaluate_fifo_relative_gap,
        paths=validation_paths,
        seed=args.seed,
        reference_env=reference_env,
        baseline_results=baseline_results,
        batch_size=args.eval_batch_size,
        evaluation_cache=validation_cache,
        restore_cache=training_cache,
    )
    callback = RCMPSPMetricsCallback(
        checkpoint_dir=args.output_dir / "checkpoints",
        early_stop_patience=args.early_stop_patience,
        validation_interval=args.validation_interval,
        validation_min_delta=args.validation_min_delta,
        validation_evaluator=validation_evaluator,
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        best_model_path = callback.best_model_path
        if best_model_path is None or not best_model_path.is_file():
            raise RuntimeError("validation did not produce a best-model checkpoint")
        model.set_parameters(best_model_path, exact_match=True, device=args.device)
        print(
            f"restored best validation model: {best_model_path}; "
            f"FIFO-relative gap={callback.best_validation_gap:+.6f}"
        )
        model.save(str(args.output_dir / "final_model"))
        result_splits = {"train": train_paths} if args.overfit_instances else splits
        result_path = evaluate_and_save_results(
            model, result_splits, args.seed, reference_env, args.output_dir,
            baseline_results,
            max_eval_instances=args.eval_max_instances,
            eval_batch_size=args.eval_batch_size,
        )
        if args.compare_with is not None:
            compare_results(
                args.compare_with,
                result_path,
                args.output_dir / "comparison.csv",
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
