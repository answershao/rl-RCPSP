#!/usr/bin/env python3
"""Evaluate the selected PPO policy with 32 sampled trajectories."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3 import PPO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_ppo import load_baseline_results, resolve_device
from src.core.rcmpsp import parse_rcmp
from src.environments.observation import build_static_graph_cache
from src.training.ppo import MAX_SAMPLE_TRAJECTORIES, evaluate_paths_sampled


SAMPLE_BUDGET = MAX_SAMPLE_TRAJECTORIES
METHOD_NAME = f"PPO-sampled-{SAMPLE_BUDGET}"
SPLITS = ("validation", "test")
MethodResults = list[tuple[str, float]]
SeedResults = dict[int, dict[str, MethodResults]]


def _evaluate_worker(
    model_path: str,
    paths: list[str],
    path_offset: int,
    seed: int,
    batch_size: int,
    max_activities: int,
    max_resources: int,
    device: str,
) -> MethodResults:
    """Load one model per process for sampled evaluation."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = PPO.load(model_path, device=device)
    try:
        instances = [parse_rcmp(path) for path in paths]
        evaluation_cache = build_static_graph_cache(
            instances,
            max_activities=max_activities,
            max_resources=max_resources,
            max_successors=model.policy.features_extractor.max_successors,
        )
        sampled = evaluate_paths_sampled(
            model,
            paths,
            seed,
            SimpleNamespace(
                max_activities=max_activities,
                max_resources=max_resources,
            ),
            sample_budgets=(SAMPLE_BUDGET,),
            batch_size=batch_size,
            evaluation_cache=evaluation_cache,
            path_offset=path_offset,
        )
        return sampled[SAMPLE_BUDGET]
    finally:
        model.policy.set_training_mode(False)


def evaluate_seed_split(
    model_path: Path,
    paths: list[str],
    seed: int,
    evaluation_seed: int,
    reference_env,
    evaluation_cache,
    batch_size: int,
    device: str,
    workers: int,
) -> MethodResults:
    """Evaluate one split with the selected sampled policy."""
    if not paths:
        return []
    if workers == 1:
        model = PPO.load(str(model_path), device=device)
        try:
            sampled = evaluate_paths_sampled(
                model,
                paths,
                evaluation_seed + seed,
                reference_env,
                sample_budgets=(SAMPLE_BUDGET,),
                batch_size=batch_size,
                evaluation_cache=evaluation_cache,
            )
            return sampled[SAMPLE_BUDGET]
        finally:
            model.policy.set_training_mode(False)
            del model

    chunk_size = (len(paths) + workers - 1) // workers
    chunks = [
        (start, paths[start : start + chunk_size])
        for start in range(0, len(paths), chunk_size)
    ]
    context = mp.get_context("spawn")
    results: MethodResults = []
    with ProcessPoolExecutor(
        max_workers=len(chunks), mp_context=context
    ) as executor:
        futures = [
            executor.submit(
                _evaluate_worker,
                str(model_path),
                chunk_paths,
                start,
                evaluation_seed + seed,
                batch_size,
                reference_env.max_activities,
                reference_env.max_resources,
                device,
            )
            for start, chunk_paths in chunks
        ]
        for future in futures:
            results.extend(future.result())
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("outputs/experiments/ppo/makespan_only"),
        help="directory containing seedN/<model-file> and seedN/splits.json",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("checkpoints/best_model.zip"),
        help="checkpoint path relative to each seed directory",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 31])
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path("outputs/baselines_mplib2_10_50_5/makespan_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/ppo/makespan_only/inference_search"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="number of instances per sampled batch",
    )
    parser.add_argument(
        "--eval-max-instances",
        type=int,
        default=0,
        help="evaluate at most N validation/test instances per split; 0 evaluates all",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="skip test evaluation and write validation results only",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel CPU workers; each worker loads its own PPO model",
    )
    parser.add_argument("--torch-threads", type=int, default=20)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=20260908)
    return parser.parse_args()


def load_split_manifest(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"split manifest not found: {path}")
    with path.open(encoding="utf-8") as stream:
        splits = json.load(stream)
    if not isinstance(splits, dict) or any(
        split not in splits or not isinstance(splits[split], list) for split in SPLITS
    ):
        raise ValueError(f"invalid split manifest: {path}")
    return splits


def load_shared_splits(models_root: Path, seeds: list[int]) -> dict[str, list[str]]:
    if not seeds:
        raise ValueError("seeds must not be empty")
    first = load_split_manifest(models_root / f"seed{seeds[0]}" / "splits.json")
    for seed in seeds[1:]:
        current = load_split_manifest(models_root / f"seed{seed}" / "splits.json")
        if current != first:
            raise ValueError("all seed split manifests must be identical")
    return first


def _aggregate(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


def calculate_metrics(
    results: MethodResults,
    baseline_results: dict[str, dict[str, int]],
) -> dict[str, float | int]:
    names = [name for name, _ in results]
    ppo = np.asarray([makespan for _, makespan in results], dtype=np.float64)
    fifo = np.asarray([baseline_results[name]["FIFO"] for name in names], dtype=np.float64)
    cp_sat = np.asarray(
        [baseline_results[name]["CP-SAT"] for name in names], dtype=np.float64
    )
    if np.any(fifo <= 0) or np.any(cp_sat <= 0):
        raise ValueError("FIFO and CP-SAT makespans must be positive")
    fifo_gap = (ppo - fifo) / fifo
    cp_sat_gap = (ppo - cp_sat) / cp_sat
    delta = ppo - fifo
    return {
        "ppo_mean": float(ppo.mean()),
        "ppo_instance_std": float(ppo.std()),
        "fifo_relative_gap": float(fifo_gap.mean()),
        "cp_sat_relative_gap": float(cp_sat_gap.mean()),
        "wins": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta > 0).sum()),
        "instances": len(results),
    }


def write_split_summary(
    path: Path,
    split: str,
    paths: list[str],
    results: MethodResults,
    baseline_results: dict[str, dict[str, int]],
) -> Path:
    fields = [
        "instance",
        "split",
        METHOD_NAME,
        "FIFO",
        "Shortest",
        "Random",
        "CP-SAT",
    ]
    values = dict(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for instance_path in paths:
            instance = Path(instance_path).name
            writer.writerow(
                {
                    "instance": instance,
                    "split": split,
                    METHOD_NAME: int(values[instance]),
                    "FIFO": baseline_results[instance]["FIFO"],
                    "Shortest": baseline_results[instance]["Shortest"],
                    "Random": baseline_results[instance]["Random"],
                    "CP-SAT": baseline_results[instance]["CP-SAT"],
                }
            )
    return path


def write_per_seed_metrics(
    path: Path,
    seed_results: SeedResults,
    baseline_results: dict[str, dict[str, int]],
) -> Path:
    fields = [
        "seed",
        "split",
        "method",
        "budget",
        "ppo_mean",
        "ppo_instance_std",
        "fifo_relative_gap",
        "cp_sat_relative_gap",
        "wins",
        "ties",
        "losses",
        "instances",
    ]
    rows = []
    for seed, split_results in seed_results.items():
        for split, results in split_results.items():
            metrics = calculate_metrics(results, baseline_results)
            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "method": METHOD_NAME,
                    "budget": SAMPLE_BUDGET,
                    **{
                        key: f"{value:.6f}" if isinstance(value, float) else value
                        for key, value in metrics.items()
                    },
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_aggregate_metrics(
    path: Path,
    seed_results: SeedResults,
    baseline_results: dict[str, dict[str, int]],
    splits: tuple[str, ...] = SPLITS,
) -> Path:
    fields = [
        "split",
        "method",
        "budget",
        "seeds",
        "ppo_mean",
        "ppo_seed_std",
        "ppo_instance_std_mean",
        "fifo_relative_gap",
        "fifo_gap_seed_std",
        "cp_sat_relative_gap",
        "cp_sat_gap_seed_std",
        "wins",
        "ties",
        "losses",
        "instances_per_seed",
    ]
    rows = []
    for split in splits:
        seed_metrics = [
            calculate_metrics(seed_results[seed][split], baseline_results)
            for seed in seed_results
            if split in seed_results[seed]
        ]
        ppo_mean, ppo_seed_std = _aggregate(
            [float(metrics["ppo_mean"]) for metrics in seed_metrics]
        )
        fifo_gap, fifo_gap_seed_std = _aggregate(
            [float(metrics["fifo_relative_gap"]) for metrics in seed_metrics]
        )
        cp_sat_gap, cp_sat_gap_seed_std = _aggregate(
            [float(metrics["cp_sat_relative_gap"]) for metrics in seed_metrics]
        )
        rows.append(
            {
                "split": split,
                "method": METHOD_NAME,
                "budget": SAMPLE_BUDGET,
                "seeds": ",".join(str(seed) for seed in seed_results),
                "ppo_mean": f"{ppo_mean:.4f}",
                "ppo_seed_std": f"{ppo_seed_std:.4f}",
                "ppo_instance_std_mean": f"{np.mean([float(metrics['ppo_instance_std']) for metrics in seed_metrics]):.4f}",
                "fifo_relative_gap": f"{fifo_gap:.6f}",
                "fifo_gap_seed_std": f"{fifo_gap_seed_std:.6f}",
                "cp_sat_relative_gap": f"{cp_sat_gap:.6f}",
                "cp_sat_gap_seed_std": f"{cp_sat_gap_seed_std:.6f}",
                "wins": sum(int(metrics["wins"]) for metrics in seed_metrics),
                "ties": sum(int(metrics["ties"]) for metrics in seed_metrics),
                "losses": sum(int(metrics["losses"]) for metrics in seed_metrics),
                "instances_per_seed": ",".join(
                    str(int(metrics["instances"])) for metrics in seed_metrics
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    args = parse_args()
    if (
        not args.seeds
        or args.eval_batch_size < 1
        or args.eval_max_instances < 0
        or args.workers < 1
        or args.torch_threads < 1
        or args.torch_interop_threads < 1
    ):
        raise ValueError(
            "seeds must be non-empty; batch size and workers must be positive"
        )

    args.device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    if args.workers > 1 and args.device != "cpu":
        raise ValueError("--workers greater than 1 requires --device cpu")

    splits = load_shared_splits(args.models_root, args.seeds)
    eval_paths = {split: list(splits[split]) for split in SPLITS}
    if args.eval_max_instances:
        eval_paths = {
            split: paths[: args.eval_max_instances]
            for split, paths in eval_paths.items()
        }
    all_paths = list(
        dict.fromkeys(path for paths in splits.values() for path in paths)
    )
    baseline_results = load_baseline_results(
        args.baseline_results, [Path(path).name for path in all_paths]
    )
    instances = {path: parse_rcmp(path) for path in all_paths}
    max_activities = max(len(instance.activities) for instance in instances.values())
    max_resources = max(instance.resource_count for instance in instances.values())
    reference_env = SimpleNamespace(
        max_activities=max_activities,
        max_resources=max_resources,
    )
    validation_cache = build_static_graph_cache(
        [instances[path] for path in eval_paths["validation"]],
        max_activities=max_activities,
        max_resources=max_resources,
    )
    test_cache = build_static_graph_cache(
        [instances[path] for path in eval_paths["test"]],
        max_activities=max_activities,
        max_resources=max_resources,
    )

    seed_results: SeedResults = {}
    for seed in args.seeds:
        model_path = args.models_root / f"seed{seed}" / args.model_file
        if not model_path.is_file():
            raise FileNotFoundError(f"PPO model not found: {model_path}")
        print(f"evaluating validation seed={seed}: {model_path}")
        seed_results[seed] = {
            "validation": evaluate_seed_split(
                model_path,
                eval_paths["validation"],
                seed,
                args.evaluation_seed,
                reference_env,
                validation_cache,
                args.eval_batch_size,
                args.device,
                args.workers,
            )
        }
        if not args.validation_only:
            print(f"evaluating test seed={seed}: {model_path}")
            seed_results[seed]["test"] = evaluate_seed_split(
                model_path,
                eval_paths["test"],
                seed,
                args.evaluation_seed,
                reference_env,
                test_cache,
                args.eval_batch_size,
                args.device,
                args.workers,
            )

    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed{seed}"
        write_split_summary(
            seed_dir / "validation_search_summary.csv",
            "validation",
            eval_paths["validation"],
            seed_results[seed]["validation"],
            baseline_results,
        )
        if not args.validation_only:
            write_split_summary(
                seed_dir / "test_search_summary.csv",
                "test",
                eval_paths["test"],
                seed_results[seed]["test"],
                baseline_results,
            )
    write_per_seed_metrics(
        args.output_dir / "per_seed_metrics.csv", seed_results, baseline_results
    )
    write_aggregate_metrics(
        args.output_dir / "aggregate.csv", seed_results, baseline_results
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "search_config.json").open(
        "w", encoding="ascii"
    ) as stream:
        json.dump(
            {
                "reward_function": "makespan_only",
                "method": METHOD_NAME,
                "sample_trajectories": SAMPLE_BUDGET,
                "model_root": str(args.models_root),
                "model_file": str(args.model_file),
                "seeds": args.seeds,
                "validation_only": args.validation_only,
                "evaluation_seed": args.evaluation_seed,
                "device": args.device,
                "eval_batch_size": args.eval_batch_size,
                "eval_max_instances": args.eval_max_instances,
                "workers": args.workers,
                "torch_threads": args.torch_threads,
                "torch_interop_threads": args.torch_interop_threads,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(f"inference search results: {args.output_dir}")


if __name__ == "__main__":
    main()
