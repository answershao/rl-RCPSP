"""Aggregate priority-rule / GA / GPHH result CSVs into one comparison table
with PSPLIB BKS gaps.

Row identity across files is the unique relative ``file`` column (e.g.
``psplib/j30/j3010_1.sm``).  When a PSPLIB BKS is known for an instance the
script appends ``bks`` and ``gap_<method>`` columns
(``100 * (makespan - bks) / bks``) and refuses to silently pass a method that
is below BKS (prints an alert row by row instead).

Outputs (under ``--out-dir``):
  merged_detail.csv     one row per instance; rules subset + GA + GPHH + BKS
  summary_by_suite.csv  per suite x method: n, mean makespan, mean gap %,
                        # instances reaching BKS, # times best among methods
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Columns kept from the full rule CSV for the detail/summary tables.
RULE_SUBSET = ["serial_FIFO", "serial_SPT", "serial_LST", "serial_LFT",
               "serial_MSLK", "serial_MTS", "serial_GRPW",
               "parallel_WCS"]
RULE_SUBSET_PREFIX = ["serial_", "parallel_"]

METHOD_ORDER = RULE_SUBSET + [
    "serial_best", "parallel_best", "rule_best", "ga_makespan", "gphh_makespan",
    "ppo_makespan",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def to_int(row: dict[str, str], col: str) -> int | None:
    v = row.get(col)
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def discover_rule_cols(rows: list[dict[str, str]]) -> list[str]:
    cols = list(rows[0].keys()) if rows else []
    return [c for c in cols if c.startswith(RULE_SUBSET_PREFIX[0]) or
            c.startswith(RULE_SUBSET_PREFIX[1])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", required=True, help="CSV from scripts/baselines.py")
    ap.add_argument("--ga", default=None, help="CSV from scripts/run_ga.py (optional)")
    ap.add_argument("--gphh", default=None, help="CSV from scripts/run_gphh.py eval_summary (optional)")
    ap.add_argument("--ppo", default=None, help="CSV from scripts/train_ppo.py ppo_eval_summary (optional)")
    ap.add_argument("--bks", default="data/bks/bks_psplib.json", help="BKS json from scripts/extract_bks.py")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import json
    bks_map = json.loads(Path(args.bks).read_text())["instances"]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    rule_rows = read_csv(Path(args.rules))
    rule_cols = discover_rule_cols(rule_rows)
    extra_best = {"serial_best": [c for c in rule_cols if c.startswith("serial_")],
                  "parallel_best": [c for c in rule_cols if c.startswith("parallel_")],
                  "rule_best": rule_cols}
    by_file: dict[str, dict] = {}
    for r in rule_rows:
        key = Path(r["file"]).as_posix()
        d = {"suite": r["suite"], "file": key,
             "n_activities": to_int(r, "n_activities"),
             "n_resources": to_int(r, "n_resources")}
        for c in rule_cols:
            d[c] = to_int(r, c)
        by_file[key] = d

    def add_source(csv_path: str | None, want_cols: dict[str, str]) -> None:
        if not csv_path:
            return
        for r in read_csv(Path(csv_path)):
            key = Path(r["file"]).as_posix()
            if key not in by_file:
                # GA/GPHH might cover instances rules did not; grow the row
                by_file[key] = {"suite": r.get("suite", ""), "file": key,
                                "n_activities": to_int(r, "n_activities"),
                                "n_resources": to_int(r, "n_resources")}
            d = by_file[key]
            for src, dst in want_cols.items():
                d[dst] = to_int(r, src)

    add_source(args.ga, {"ga_makespan": "ga_makespan"})
    add_source(args.gphh, {"gphh_makespan": "gphh_makespan"})
    add_source(args.ppo, {"ppo_makespan": "ppo_makespan"})

    # Derive best-of-family columns per row, then attach BKS + gaps.
    detail: list[dict] = []
    alerts = 0
    for d in sorted(by_file.values(), key=lambda x: (x["suite"], x["file"])):
        for name, cols in extra_best.items():
            vals = [d[c] for c in cols if d.get(c) is not None]
            d[name] = min(vals) if vals else None
        base = Path(d["file"]).name.lower()
        d["bks"] = bks_map.get(base)
        if d["bks"] is not None:
            for m in METHOD_ORDER:
                mk = d.get(m)
                if mk is None:
                    continue
                gap = 100.0 * (mk - d["bks"]) / d["bks"]
                d[f"gap_{m}"] = round(gap, 2)
                if mk < d["bks"]:
                    alerts += 1
                    print(f"ALERT below-BKS {d['file']} {m}={mk} < bks={d['bks']}")
        detail.append(d)

    detail_cols = ["suite", "file", "n_activities", "n_resources", "bks", *METHOD_ORDER]
    gap_cols = [f"gap_{m}" for m in METHOD_ORDER]
    with open(Path(args.out_dir) / "merged_detail.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=detail_cols + gap_cols, extrasaction="ignore")
        w.writeheader()
        for d in detail:
            w.writerow(d)

    # Summary per suite x method.
    suite_methods: dict[tuple[str, str], list[int]] = {}
    suite_best = {d["file"]: min((d[m] for m in METHOD_ORDER if d.get(m) is not None), default=None)
                  for d in detail}
    for d in detail:
        for m in METHOD_ORDER:
            mk = d.get(m)
            if mk is None:
                continue
            suite_methods.setdefault((d["suite"], m), []).append(
                {"mk": mk, "bks": d.get("bks"), "best": suite_best[d["file"]]})

    with open(Path(args.out_dir) / "summary_by_suite.csv", "w", newline="") as fh:
        fieldnames = ["suite", "method", "n", "mean_makespan", "mean_gap_pct",
                      "n_at_bks", "n_best_of_methods"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for (suite, method), rows in sorted(suite_methods.items()):
            mks = [r["mk"] for r in rows]
            has_bks = [r for r in rows if r["bks"] is not None]
            gaps = [100.0 * (r["mk"] - r["bks"]) / r["bks"] for r in has_bks]
            row = {
                "suite": suite, "method": method, "n": len(mks),
                "mean_makespan": round(sum(mks) / len(mks), 2),
                "mean_gap_pct": round(sum(gaps) / len(gaps), 2) if gaps else "",
                "n_at_bks": sum(1 for r in has_bks if r["mk"] <= r["bks"]),
                "n_best_of_methods": sum(1 for r in rows if r["mk"] == r["best"]),
            }
            w.writerow(row)

    print(f"instances: {len(detail)}  rule cols: {len(rule_cols)}  below-BKS alerts: {alerts}")
    print(f"wrote {Path(args.out_dir) / 'merged_detail.csv'}")
    print(f"wrote {Path(args.out_dir) / 'summary_by_suite.csv'}")


if __name__ == "__main__":
    main()
