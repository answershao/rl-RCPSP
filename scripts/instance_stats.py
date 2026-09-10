"""Characterise the instance-space coverage of the training pool vs the eval suites.

Motivation
----------
The PPO policy is trained on the generated pool but evaluated on PSPLIB
j30-j120. Whether that transfer can work at all depends on how well the
training pool covers the *input distribution the policy actually consumes* --
not on raw instance counts.  This script measures that, so any change to the
protocol (new generator settings, re-weighting, different normalisation) can be
re-checked instead of argued about.

It reports three layers:

1. **Instance parameters** -- n, K, RF (resource factor), RS (resource
   strength), NC (network complexity), CP, resource lower bound.
2. **Policy input features** -- the normalised per-node tensors built by
   ``src.envs.observation``: ``duration / max(duration)``,
   ``downstream / max(downstream)``, ``demand / capacity``, ``degree / MAX_*``.
   These are what the GIN sees.
3. **Coverage** -- for each evaluation suite, the fraction of instances whose
   (RF, RS) fall inside the training-pool envelope, plus a histogram-intersection
   overlap of every policy input feature against the training pool.

Usage:
    python -m scripts.instance_stats --data-root data --splits splits.json \
        --output-dir outputs/instance_stats --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scripts.common import map_jobs
from src.data.adapter import load_core_instance
from src.data.instances import read_protocol

# Must match src.envs.observation; duplicated on purpose so this diagnostic can
# run without importing the torch stack.
MAX_SUCCESSORS = 20
MAX_PREDECESSORS = 20

FEATURE_COLUMNS = (
    "dur_over_max_duration",
    "downstream_over_max_downstream",
    "demand_over_capacity",
    "outdeg_norm",
    "indeg_norm",
)
PARAM_COLUMNS = ("n", "K", "RF", "RS", "NC", "OS", "CP", "sumdur", "res_lb", "res_dom")
FIELDS = ("group", "file", *PARAM_COLUMNS, *FEATURE_COLUMNS)


def instance_parameters(inst) -> dict:
    """Standard RCPSP descriptors for one parsed instance.

    Shared by this diagnostic and ``scripts/generate_pool.py`` so a generated
    pool is measured with exactly the same definitions as a benchmark suite.

    * ``RF`` -- mean fraction of resource types each real activity consumes
    * ``RS`` -- mean over resources of ``(a_k - r_min_k) / (r_max_k - r_min_k)``
      with ``r_min_k = max_i d_ik`` and ``r_max_k = sum_i d_ik`` (lower = tighter)
    * ``NC`` -- arcs per node, counted over all nodes including the dummies
    * ``OS`` -- order strength: precedence-feasible pairs in the transitive
      closure over real activities, divided by ``C(n_real, 2)``.  This is the
      RCPLIB workbook's definition (verified against it); the direct-arc ratio
      would be roughly three times smaller and is NOT what PSPLIB reports.
    """
    ids = sorted(inst.activities)
    durations = np.array([inst.activities[i].duration for i in ids], dtype=float)
    demands = np.array([inst.activities[i].demand for i in ids], dtype=float)
    capacities = np.maximum(np.asarray(inst.capacities, dtype=float), 1.0)

    real = durations > 0
    n_real = int(real.sum())
    k = len(capacities)
    if n_real == 0:
        raise ValueError(f"{inst.name}: instance has no real activities")
    dur_r = durations[real]
    dem_r = demands[real]

    rf = float((dem_r > 0).sum(axis=1).mean() / k) if k else 0.0
    rs_terms = []
    for r in range(k):
        column = dem_r[:, r]
        r_max = float(column.sum())
        r_min = float(column.max())
        rs_terms.append(
            (capacities[r] - r_min) / (r_max - r_min) if r_max > r_min else 1.0
        )
    rs = float(np.mean(rs_terms)) if rs_terms else 1.0

    successors = {i: inst.activities[i].successors for i in ids}
    predecessors = {i: inst.predecessors.get(i, ()) for i in ids}
    arcs = sum(len(successors[i]) for i in ids)
    nc = arcs / n_real

    order: list[int] = []
    indeg = {i: len(predecessors[i]) for i in ids}
    queue = [i for i in ids if indeg[i] == 0]
    while queue:
        node = queue.pop()
        order.append(node)
        for nxt in successors[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(ids):
        raise ValueError(f"{inst.name}: precedence graph contains a cycle")
    downstream = np.zeros(len(ids), dtype=float)
    for node in reversed(order):
        downstream[node] = durations[node] + max(
            (downstream[nxt] for nxt in successors[node]), default=0.0
        )
    cp = float(downstream.max())

    # Order strength over the transitive closure, as a bit mask per node.
    index_of = {node: i for i, node in enumerate(ids)}
    real_mask = 0
    for i, node in enumerate(ids):
        if durations[i] > 0:
            real_mask |= 1 << i
    descendants = [0] * len(ids)
    for node in reversed(order):  # topological: successors are already done
        mask = 0
        for nxt in successors[node]:
            j = index_of[nxt]
            mask |= (1 << j) | descendants[j]
        descendants[index_of[node]] = mask
    reachable_pairs = sum(
        bin(descendants[index_of[node]] & real_mask).count("1")
        for i, node in enumerate(ids)
        if durations[i] > 0
    )
    os_ = (
        reachable_pairs / (n_real * (n_real - 1) / 2) if n_real > 1 else 0.0
    )

    res_lb = 0.0
    for r in range(k):
        workload = float((dem_r[:, r] * dur_r).sum())
        res_lb = max(res_lb, math.ceil(workload / capacities[r]))

    return {
        "n": float(n_real),
        "K": float(k),
        "RF": rf,
        "RS": rs,
        "NC": nc,
        "OS": float(os_),
        "CP": cp,
        "sumdur": float(dur_r.sum()),
        "res_lb": float(res_lb),
        "res_dom": float(res_lb) / cp if cp > 0 else 0.0,
    }


def instance_descriptors(path: Path, group: str, rel: str) -> dict:
    """Compute every parameter / policy-input descriptor for one instance.

    ``_nodes`` holds per-activity arrays used for the feature-overlap analysis;
    it is stripped before the row is written to CSV.
    """
    inst = load_core_instance(path)
    row = {"group": group, "file": rel, **instance_parameters(inst)}

    ids = sorted(inst.activities)
    durations = np.array([inst.activities[i].duration for i in ids], dtype=float)
    demands = np.array([inst.activities[i].demand for i in ids], dtype=float)
    capacities = np.maximum(np.asarray(inst.capacities, dtype=float), 1.0)
    real = durations > 0
    # Mirror src/envs/observation.py: static duration features are normalised
    # by per-instance maxima (size invariant), not by the duration sum.
    duration_scale = max(float(durations[real].max()) if real.any() else 1.0, 1.0)

    order: list[int] = []
    indeg = {i: len(inst.predecessors.get(i, ())) for i in ids}
    queue = [i for i in ids if indeg[i] == 0]
    while queue:
        node = queue.pop()
        order.append(node)
        for nxt in inst.activities[node].successors:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    downstream = np.zeros(len(ids), dtype=float)
    for node in reversed(order):
        downstream[node] = durations[node] + max(
            (
                downstream[nxt]
                for nxt in inst.activities[node].successors
            ),
            default=0.0,
        )

    downstream_scale = max(float(downstream.max()), 1.0)

    row["_nodes"] = {
        "dur_over_max_duration": durations[real] / duration_scale,
        "downstream_over_max_downstream": downstream[real] / downstream_scale,
        "demand_over_capacity": (demands[real] / capacities).ravel(),
        "outdeg_norm": np.array(
            [len(inst.activities[i].successors) for i in ids], dtype=float
        )[real]
        / MAX_SUCCESSORS,
        "indeg_norm": np.array(
            [len(inst.predecessors.get(i, ())) for i in ids], dtype=float
        )[real]
        / MAX_PREDECESSORS,
    }
    return row


def collect_jobs(
    protocol: dict, data_root: Path, extra_pools: Sequence[tuple[str, str]] = ()
) -> list[tuple[Path, str, str]]:
    """Build ``(path, group, relative-key)`` jobs for the protocol plus any extras.

    ``extra_pools`` entries are ``(group_name, glob)`` pairs resolved against
    ``--data-root``, which is how a generated candidate pool gets measured with
    the same code path as a benchmark suite.
    """
    jobs = [(data_root / rel, "train", rel) for rel in protocol["train"]]
    jobs += [(data_root / rel, "validation", rel) for rel in protocol["validation"]]
    for suite, rels in protocol["evaluation"].items():
        jobs += [(data_root / rel, f"eval_{suite}", rel) for rel in rels]
    for group, pattern in extra_pools:
        paths = sorted(data_root.glob(pattern))
        if not paths:
            raise ValueError(f"extra pool {group!r} matched no files under {data_root}/{pattern}")
        jobs += [(path, group, path.relative_to(data_root).as_posix()) for path in paths]
    return jobs


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    out = []
    for group, members in groups.items():
        entry: dict = {"group": group, "n_instances": len(members)}
        for column in (*PARAM_COLUMNS, *FEATURE_COLUMNS):
            values = np.array([m[column] for m in members], dtype=float)
            entry[f"{column}_mean"] = float(values.mean())
            entry[f"{column}_std"] = float(values.std())
            entry[f"{column}_min"] = float(values.min())
            entry[f"{column}_max"] = float(values.max())
        out.append(entry)
    out.sort(key=lambda e: e["group"])
    return out


def feature_means(row: dict) -> dict[str, float]:
    return {name: float(np.mean(values)) for name, values in row["_nodes"].items()}


def overlap(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    """Histogram intersection of two 1-D samples, in [0, 1]."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    h1, _ = np.histogram(a, bins=edges)
    h2, _ = np.histogram(b, bins=edges)
    h1 = h1 / max(h1.sum(), 1)
    h2 = h2 / max(h2.sum(), 1)
    return float(np.minimum(h1, h2).sum())


def coverage_rows(rows: list[dict], rf_tol: float) -> list[dict]:
    train = [r for r in rows if r["group"] == "train"]
    rf_lo = min(r["RF"] for r in train) - rf_tol
    rf_hi = max(r["RF"] for r in train) + rf_tol
    rs_lo = min(r["RS"] for r in train)
    rs_hi = max(r["RS"] for r in train)
    pool: dict[str, list[np.ndarray]] = {
        name: [r["_nodes"][name] for r in train] for name in FEATURE_COLUMNS
    }
    pool = {name: np.concatenate(values) for name, values in pool.items()}

    out = []
    for group in sorted({r["group"] for r in rows}):
        members = [r for r in rows if r["group"] == group]
        entry: dict = {
            "group": group,
            "n_instances": len(members),
            "train_RF_lo": rf_lo,
            "train_RF_hi": rf_hi,
            "train_RS_lo": rs_lo,
            "train_RS_hi": rs_hi,
        }
        entry["RF_inside_pct"] = 100.0 * float(
            np.mean([rf_lo <= r["RF"] <= rf_hi for r in members])
        )
        entry["RS_inside_pct"] = 100.0 * float(
            np.mean([rs_lo <= r["RS"] <= rs_hi for r in members])
        )
        entry["joint_inside_pct"] = 100.0 * float(
            np.mean(
                [
                    rf_lo <= r["RF"] <= rf_hi and rs_lo <= r["RS"] <= rs_hi
                    for r in members
                ]
            )
        )
        for name in FEATURE_COLUMNS:
            sample = np.concatenate([r["_nodes"][name] for r in members])
            entry[f"overlap_{name}"] = overlap(pool[name], sample)
        out.append(entry)
    return out


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in fields} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits", type=Path, default=Path("splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/instance_stats"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--rf-tol",
        type=float,
        default=0.05,
        help="RF tolerance when testing whether an instance sits inside the pool",
    )
    parser.add_argument(
        "--extra-pool",
        action="append",
        default=[],
        metavar="NAME=GLOB",
        help="measure an additional instance pool, e.g. "
        "'gen_grid=generated/grid/**/*.rcp'; repeatable",
    )
    args = parser.parse_args()

    extra_pools: list[tuple[str, str]] = []
    for item in args.extra_pool:
        if "=" not in item:
            raise ValueError(f"--extra-pool expects NAME=GLOB, got {item!r}")
        name, pattern = item.split("=", 1)
        extra_pools.append((name, pattern))

    protocol = read_protocol(args.splits)
    jobs = collect_jobs(protocol, args.data_root, extra_pools)
    rows = map_jobs(instance_descriptors, jobs, args.workers)
    for row in rows:
        row.update(feature_means(row))
    rows.sort(key=lambda r: (r["group"], r["file"]))

    summary = summarise(rows)
    coverage = coverage_rows(rows, args.rf_tol)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "instances.csv", list(FIELDS))
    write_csv(summary, args.output_dir / "summary.csv", list(summary[0]))
    write_csv(coverage, args.output_dir / "coverage.csv", list(coverage[0]))
    (args.output_dir / "coverage.json").write_text(json.dumps(coverage, indent=1))

    print(f"{'group':<18}{'N':>5}" + "".join(f"{c:>13}" for c in PARAM_COLUMNS))
    for entry in summary:
        line = f"{entry['group']:<18}{entry['n_instances']:>5}"
        for column in PARAM_COLUMNS:
            line += f"{entry[f'{column}_mean']:>13.3f}"
        print(line)

    print()
    print(f"{'group':<18}{'RF in':>8}{'RS in':>8}{'joint':>8}"
          + "".join(f"{c[:12]:>14}" for c in FEATURE_COLUMNS))
    for entry in coverage:
        line = (f"{entry['group']:<18}{entry['RF_inside_pct']:>7.1f}%"
                f"{entry['RS_inside_pct']:>7.1f}%{entry['joint_inside_pct']:>7.1f}%")
        for column in FEATURE_COLUMNS:
            line += f"{entry[f'overlap_{column}']:>14.3f}"
        print(line)
    print(f"\nwrote {args.output_dir}/instances.csv, summary.csv, coverage.csv")


if __name__ == "__main__":
    main()
