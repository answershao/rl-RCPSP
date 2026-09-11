#!/usr/bin/env python3
"""Evaluate selected PPO checkpoints with 32 sampled trajectories per seed.

This is the post-training inference search over the fixed RCPSP protocol:

* checkpoint layouts: legacy ``<models-root>/seed<N>/<model-file>`` or the
  newest timestamped CPU run ``<models-root>/cpu_YYYYMMDD_HHMMSS/<model-file>``;
* evaluated instance groups come from ``splits.json`` (``validation`` plus
  any ``evaluation`` group such as ``psplib_j30``), shared by every seed;
* instances load through ``src.data.adapter`` and names are the unique
  data-root-relative ids, so results join cleanly with the baseline CSVs.

Metrics are reported per group and aggregated across seeds; when ``--ref-rules``
(a ``scripts/baselines.py`` CSV) is given, each run is also expressed as a
rule-relative gap (default reference column ``serial_LST``).
"""

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

from scripts.train_ppo import resolve_device
from scripts.ppo_runs import latest_training_run, resolve_model_path
from src.data.instances import instance_id, loader_for, read_protocol
from src.training.ppo import (
    MAX_SAMPLE_TRAJECTORIES,
    evaluate_paths,
    evaluate_paths_sampled,
)

SAMPLE_BUDGET = MAX_SAMPLE_TRAJECTORIES
METHOD_NAME = f"PPO-sampled-{SAMPLE_BUDGET}"
DEFAULT_MODEL_FILE = Path("checkpoints/best_model.zip")
MethodResults = list[tuple[str, float]]
SeedResults = dict[int, dict[str, MethodResults]]


def resolve_search_model_paths(
    models_root: Path, model_file: Path, seeds: list[int]
) -> dict[int, Path]:
    """Select one checkpoint for every seed used by the search.

    The normal search target is always the best checkpoint from the newest
    timestamped PPO run.  Other model files remain available as an explicit
    compatibility override for legacy seed layouts.
    """
    if model_file == DEFAULT_MODEL_FILE:
        run_dir = latest_training_run(models_root, DEFAULT_MODEL_FILE)
        model_path = run_dir / DEFAULT_MODEL_FILE
        return {seed: model_path for seed in seeds}
    return {
        seed: resolve_model_path(models_root, model_file, seed)
        for seed in seeds
    }


def _reference_env_for(model: PPO) -> SimpleNamespace:
    extractor = model.policy.features_extractor
    return SimpleNamespace(
        max_activities=extractor.max_activities,
        max_resources=extractor.max_resources,
        max_horizon=extractor.max_horizon,
    )


def _evaluate_worker(
    model_path: str,
    rels: list[str],
    seed: int,
    batch_size: int,
    device: str,
    loader,
    name_fn,
    path_offset: int,
) -> tuple[MethodResults, MethodResults]:
    """Load one model per process for deterministic and sampled evaluation."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = PPO.load(model_path, device=device)
    try:
        deterministic = evaluate_paths(
            model,
            rels,
            seed,
            _reference_env_for(model),
            batch_size=batch_size,
            loader=loader,
            name_fn=name_fn,
        )
        sampled = evaluate_paths_sampled(
            model,
            rels,
            seed,
            _reference_env_for(model),
            sample_budgets=(SAMPLE_BUDGET,),
            batch_size=batch_size,
            path_offset=path_offset,
            loader=loader,
            name_fn=name_fn,
        )
        return deterministic, sampled[SAMPLE_BUDGET]
    finally:
        model.policy.set_training_mode(False)


def evaluate_group(
    model_path: Path,
    rels: list[str],
    seed: int,
    batch_size: int,
    device: str,
    loader,
    name_fn,
    workers: int,
    evaluation_seed: int,
) -> tuple[MethodResults, MethodResults]:
    """Evaluate one group with deterministic and sampled policies."""
    if not rels:
        return [], []
    if workers == 1:
        model = PPO.load(str(model_path), device=device)
        try:
            deterministic = evaluate_paths(
                model,
                rels,
                evaluation_seed + seed,
                _reference_env_for(model),
                batch_size=batch_size,
                loader=loader,
                name_fn=name_fn,
            )
            sampled = evaluate_paths_sampled(
                model,
                rels,
                evaluation_seed + seed,
                _reference_env_for(model),
                sample_budgets=(SAMPLE_BUDGET,),
                batch_size=batch_size,
                loader=loader,
                name_fn=name_fn,
            )
            return deterministic, sampled[SAMPLE_BUDGET]
        finally:
            model.policy.set_training_mode(False)
            del model

    chunk_size = (len(rels) + workers - 1) // workers
    chunks = [
        (start, rels[start : start + chunk_size])
        for start in range(0, len(rels), chunk_size)
    ]
    context = mp.get_context("spawn")
    deterministic_results: MethodResults = []
    sampled_results: MethodResults = []
    with ProcessPoolExecutor(
        max_workers=len(chunks), mp_context=context
    ) as executor:
        futures = [
            executor.submit(
                _evaluate_worker,
                str(model_path),
                chunk_rels,
                evaluation_seed + seed,
                batch_size,
                device,
                loader,
                name_fn,
                start,
            )
            for start, chunk_rels in chunks
        ]
        for future in futures:
            deterministic, sampled = future.result()
            deterministic_results.extend(deterministic)
            sampled_results.extend(sampled)
    return deterministic_results, sampled_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("outputs/experiments/ppo/cpu_runs"),
        help="directory containing seedN/<model-file> or timestamped CPU runs",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help="checkpoint override; the default selects best_model.zip from the newest PPO run",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--splits", type=Path, default=Path("splits.json"),
        help="protocol file generated by scripts/generate_pool.py",
    )
    parser.add_argument(
        "--eval-groups",
        default="psplib_j30,psplib_j60,psplib_j90,psplib_j120",
        help="comma-separated groups to evaluate: any evaluation group id from "
             "splits.json and/or the 'validation' split",
    )
    parser.add_argument(
        "--ref-rules",
        type=Path,
        default=Path("outputs/rules_psplib/makespan_summary.csv"),
        help="baselines.py CSV covering the evaluated groups (optional)",
    )
    parser.add_argument(
        "--ref-rule",
        default="serial_LST",
        help="column of --ref-rules used for the rule-relative gap",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="search output directory; defaults to the selected run/inference_search",
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
        help="evaluate at most N instances per group; 0 evaluates all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel CPU workers; each worker loads its own PPO model",
    )
    parser.add_argument("--torch-threads", type=int, default=20)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=20260910)
    return parser.parse_args()


def load_reference_rules(path: Path, column: str) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"--ref-rules not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if column not in (reader.fieldnames or ()):
            raise ValueError(f"{path} is missing reference column {column!r}")
        refs: dict[str, int] = {}
        for raw in reader:
            key = instance_id(raw["file"])
            try:
                refs[key] = int(float(raw[column]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid {column} for {raw.get('file')!r}") from exc
    return refs


def _aggregate(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    )


def calculate_metrics(
    results: MethodResults,
    reference_rules: dict[str, int],
) -> dict[str, float | int]:
    names = [name for name, _ in results]
    ppo = np.asarray([makespan for _, makespan in results], dtype=np.float64)
    rule = np.asarray([reference_rules[name] for name in names], dtype=np.float64)
    if np.any(rule <= 0):
        raise ValueError("reference rule makespans must be positive")
    rule_gap = (ppo - rule) / rule
    delta = ppo - rule
    return {
        "ppo_mean": float(ppo.mean()),
        "ppo_instance_std": float(ppo.std()),
        "rule_relative_gap": float(rule_gap.mean()),
        "wins": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta > 0).sum()),
        "instances": len(results),
    }


def print_comparison(
    group: str,
    deterministic_results: list[MethodResults],
    sampled_results: list[MethodResults],
    reference_rules: dict[str, int] | None,
) -> None:
    """Print aggregate deterministic/sampled results for one evaluation group."""
    deterministic_values = [
        makespan
        for seed_results in deterministic_results
        for _, makespan in seed_results
    ]
    sampled = [
        result
        for seed_results in sampled_results
        for result in seed_results
    ]
    if not deterministic_values or not sampled:
        return
    deterministic_mean = float(np.mean(deterministic_values))
    sampled_mean = float(np.mean([makespan for _, makespan in sampled]))
    if reference_rules is None:
        print(
            f"{group}: deterministic makespan mean={deterministic_mean:.4f}; "
            f"sampled-32 makespan mean={sampled_mean:.4f}"
        )
        return
    reference_values = [reference_rules[name] for name, _ in sampled]
    reference_mean = float(np.mean(reference_values))
    metrics = calculate_metrics(sampled, reference_rules)
    print(
        f"{group}: deterministic makespan mean={deterministic_mean:.4f}; "
        f"sampled-32 makespan mean={sampled_mean:.4f}; "
        f"serial_LST mean={reference_mean:.4f}"
    )
    print(
        f"{group}: sampled-32 vs serial_LST: "
        f"wins={metrics['wins']}, ties={metrics['ties']}, "
        f"losses={metrics['losses']}"
    )


def write_group_summary(
    path: Path,
    group: str,
    rels: list[str],
    results: MethodResults,
    reference_rules: dict[str, int] | None,
) -> Path:
    values = dict(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["instance", "group", METHOD_NAME, "serial_LST"]
            if reference_rules
            else ["instance", "group", METHOD_NAME],
        )
        writer.writeheader()
        for rel in rels:
            name = instance_id(rel)
            row = {
                "instance": name,
                "group": group,
                METHOD_NAME: int(values[name]),
            }
            if reference_rules:
                row["serial_LST"] = reference_rules.get(name, "")
            writer.writerow(row)
    return path


def write_per_seed_metrics(
    path: Path,
    seed_results: SeedResults,
    groups: list[str],
    reference_rules: dict[str, int] | None,
) -> Path:
    fields = ["seed", "group", "method", "budget", "ppo_mean", "ppo_instance_std"]
    if reference_rules:
        fields += ["rule_relative_gap", "wins", "ties", "losses"]
    fields += ["instances"]
    rows = []
    for seed, group_results in seed_results.items():
        for group in groups:
            results = group_results.get(group, [])
            if not results:
                continue
            metrics = calculate_metrics(results, reference_rules) if reference_rules else {
                "ppo_mean": float(np.mean([m for _, m in results])),
                "ppo_instance_std": float(np.std([m for _, m in results])),
                "instances": len(results),
            }
            rows.append(
                {
                    "seed": seed,
                    "group": group,
                    "method": METHOD_NAME,
                    "budget": SAMPLE_BUDGET,
                    **{
                        key: f"{value:.6f}" if isinstance(value, float) else value
                        for key, value in metrics.items()
                    },
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_aggregate_metrics(
    path: Path,
    seed_results: SeedResults,
    groups: list[str],
    reference_rules: dict[str, int] | None,
) -> Path:
    fields = [
        "group", "method", "budget", "seeds", "ppo_mean", "ppo_seed_std",
        "ppo_instance_std_mean",
    ]
    if reference_rules:
        fields += ["rule_relative_gap", "rule_gap_seed_std", "wins", "ties", "losses"]
    fields += ["instances_per_seed"]
    rows = []
    for group in groups:
        seed_metrics = []
        for seed in seed_results:
            results = seed_results[seed].get(group, [])
            if not results:
                continue
            seed_metrics.append(
                calculate_metrics(results, reference_rules) if reference_rules else {
                    "ppo_mean": float(np.mean([m for _, m in results])),
                    "ppo_instance_std": float(np.std([m for _, m in results])),
                    "instances": len(results),
                }
            )
        if not seed_metrics:
            continue
        ppo_mean, ppo_seed_std = _aggregate(
            [float(metrics["ppo_mean"]) for metrics in seed_metrics]
        )
        row: dict[str, object] = {
            "group": group,
            "method": METHOD_NAME,
            "budget": SAMPLE_BUDGET,
            "seeds": ",".join(str(seed) for seed in seed_results),
            "ppo_mean": f"{ppo_mean:.4f}",
            "ppo_seed_std": f"{ppo_seed_std:.4f}",
            "ppo_instance_std_mean": (
                f"{np.mean([float(metrics['ppo_instance_std']) for metrics in seed_metrics]):.4f}"
            ),
        }
        if reference_rules:
            gap, gap_std = _aggregate(
                [float(metrics["rule_relative_gap"]) for metrics in seed_metrics]
            )
            row.update(
                {
                    "rule_relative_gap": f"{gap:.6f}",
                    "rule_gap_seed_std": f"{gap_std:.6f}",
                    "wins": sum(int(metrics["wins"]) for metrics in seed_metrics),
                    "ties": sum(int(metrics["ties"]) for metrics in seed_metrics),
                    "losses": sum(int(metrics["losses"]) for metrics in seed_metrics),
                }
            )
        row["instances_per_seed"] = ",".join(
            str(int(metrics["instances"])) for metrics in seed_metrics
        )
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
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

    protocol = read_protocol(args.splits)
    available = dict(protocol["evaluation"])
    available["validation"] = protocol["validation"]
    requested = [group for group in args.eval_groups.split(",") if group]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"unknown eval groups: {unknown}; choose from {sorted(available)}")
    eval_groups = {group: list(available[group]) for group in requested}
    if args.eval_max_instances:
        eval_groups = {
            group: rels[: args.eval_max_instances] for group, rels in eval_groups.items()
        }

    loader = loader_for(args.data_root)
    name_fn = instance_id
    reference_rules = None
    if args.ref_rules is not None:
        reference_rules = load_reference_rules(args.ref_rules, args.ref_rule)
        missing = [
            rel
            for rels in eval_groups.values()
            for rel in rels
            if name_fn(rel) not in reference_rules
        ]
        if missing:
            raise ValueError(
                f"--ref-rules does not cover {len(missing)} evaluation instances; "
                f"first missing: {missing[0]}"
            )

    model_paths = resolve_search_model_paths(
        args.models_root, args.model_file, args.seeds
    )
    for seed, model_path in model_paths.items():
        print(f"selected model for seed={seed}: {model_path}")

    if args.output_dir is None:
        selected_paths = set(model_paths.values())
        if len(selected_paths) == 1:
            selected_model = next(iter(selected_paths))
            if selected_model.parent.name == "checkpoints":
                selected_run = selected_model.parent.parent
            else:
                selected_run = selected_model.parent
            args.output_dir = selected_run / "inference_search"
        else:
            args.output_dir = args.models_root / "inference_search"

    seed_results: SeedResults = {}
    deterministic_seed_results: SeedResults = {}
    for seed in args.seeds:
        model_path = model_paths[seed]
        print(f"evaluating seed={seed}: {model_path}")
        seed_results[seed] = {}
        deterministic_seed_results[seed] = {}
        for group, rels in eval_groups.items():
            if not rels:
                continue
            print(f"  group {group} (n={len(rels)})")
            deterministic, sampled = evaluate_group(
                model_path,
                rels,
                seed,
                args.eval_batch_size,
                args.device,
                loader,
                name_fn,
                args.workers,
                args.evaluation_seed,
            )
            deterministic_seed_results[seed][group] = deterministic
            seed_results[seed][group] = sampled

    groups = [group for group in eval_groups if any(group in seed_results[s] for s in args.seeds)]
    for group in groups:
        print_comparison(
            group,
            [deterministic_seed_results[seed].get(group, []) for seed in args.seeds],
            [seed_results[seed].get(group, []) for seed in args.seeds],
            reference_rules,
        )
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed{seed}"
        for group in groups:
            if group not in seed_results[seed]:
                continue
            write_group_summary(
                seed_dir / f"{group}_search_summary.csv",
                group,
                eval_groups[group],
                seed_results[seed][group],
                reference_rules,
            )
    write_per_seed_metrics(
        args.output_dir / "per_seed_metrics.csv", seed_results, groups, reference_rules
    )
    write_aggregate_metrics(
        args.output_dir / "aggregate.csv", seed_results, groups, reference_rules
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "search_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "protocol": str(args.splits),
                "data_root": str(args.data_root),
                "method": METHOD_NAME,
                "sample_trajectories": SAMPLE_BUDGET,
                "model_root": str(args.models_root),
                "model_file": str(args.model_file),
                "resolved_models": {
                    str(seed): str(model_paths[seed]) for seed in args.seeds
                },
                "seeds": args.seeds,
                "groups": groups,
                "reference_rules": str(args.ref_rules) if reference_rules else None,
                "reference_rule": args.ref_rule if reference_rules else None,
                "evaluation_seed": args.evaluation_seed,
                "device": args.device,
                "eval_batch_size": args.eval_batch_size,
                "eval_max_instances": args.eval_max_instances,
                "workers": args.workers,
                "torch_threads": args.torch_threads,
                "torch_interop_threads": args.torch_interop_threads,
                "output_dir": str(args.output_dir),
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(f"inference search results: {args.output_dir}")


if __name__ == "__main__":
    main()
