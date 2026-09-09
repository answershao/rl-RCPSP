"""Extract PSPLIB best-known solutions (BKS) from the RCPLIB parameter workbook.

The workbook has no single "BKS" column for PSPLIB. For each instance the
best known solution is taken as the minimum over every solution-side column:

  * UB-lit (literature upper bound)
  * minGA/minSS/minEM-500Kx100 (best-of-runs columns)
  * every "UB-*" column (BnB / UAB / UAA / USB / USA / UPB / UPA / LAB / LAA /
    LSB / LSA / LPB / LPA, at 1s / 1m / 1h budgets)
  * every "MH-*" column (metaheuristic upper bounds at 1m / 10m / 1h)

Output: data/bks/bks_psplib.json  {instance_name_lower: int_bks}
The console prints a per-size consistency report (how often all 1h UB methods
agree, which acts as a sanity check for the synthesized value).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from openpyxl import load_workbook

UB_NAME = re.compile(r"^(UB-|MH-)|^min(?:GA|SS|EM)-500Kx100$|^UB-lit$")


def is_ub_column(name: str) -> bool:
    return bool(UB_NAME.match(name)) if name else False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", default="data/bks/RCPLIB (Parameters and BKS).xlsx")
    ap.add_argument("--sheet", default="PSPLIB")
    ap.add_argument("--output", default="data/bks/bks_psplib.json")
    args = ap.parse_args()

    wb = load_workbook(args.workbook, read_only=True, data_only=True)
    ws = wb[args.sheet]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    name_idx = header.index("FileName")
    act_idx = header.index("#Act")
    set_idx = header.index("SubSet")
    ub_cols = [(i, str(h)) for i, h in enumerate(header) if is_ub_column(str(h))]
    ub_idx = [i for i, _ in ub_cols]
    print(f"UB columns used ({len(ub_cols)}): {[h for _, h in ub_cols]}")

    by_size: dict[str, list[dict]] = {}
    missing = 0
    for row in rows:
        fname = str(row[name_idx]).strip().lower()
        subset = str(row[set_idx]).strip()
        vals = [row[i] for i in ub_idx]
        nums = [int(v) for v in vals if isinstance(v, (int, float)) and v is not None and float(v) > 0]
        if not nums:
            missing += 1
            continue
        bks = min(nums)
        n_methods_agree = sum(1 for v in nums if v == bks)
        by_size.setdefault(subset, []).append(
            {"file": fname, "bks": bks, "candidates": len(nums), "agree_min": n_methods_agree}
        )

    instances = {}
    print(f"\n{'size':<6}{'n':>6}{'all-cand?':>10}{'agree=min':>12}{'bks mean':>10}")
    for size in sorted(by_size):
        recs = by_size[size]
        complete = sum(1 for r in recs if r["candidates"] == len(ub_idx))
        agree = sum(1 for r in recs if r["agree_min"] == r["candidates"])
        mean_bks = sum(r["bks"] for r in recs) / len(recs)
        print(f"{size:<6}{len(recs):>6}{complete:>10}{agree:>12}{mean_bks:>10.1f}")
        for r in recs[:3]:
            print(f"    {r['file']:<16} bks={r['bks']}  candidates={r['candidates']} agree_min={r['agree_min']}")
            instances[r["file"]] = r["bks"]
        for r in recs[3:]:
            instances[r["file"]] = r["bks"]

    print(f"\nmissing-candidates rows: {missing}")
    print(f"instances extracted: {len(instances)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rule": "min over " + ";".join(h for _, h in ub_cols),
                               "sheet": args.sheet, "instances": instances}, indent=0))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
