#!/usr/bin/env python3
"""Summarize an MPLIB2 directory.

Examples:
    python3 scripts/summarize_mplib2.py data/MPLIB2
    python3 scripts/summarize_mplib2.py data/MPLIB2 --json summary.json
    python3 scripts/summarize_mplib2.py data/MPLIB2 --set 3 --output summary.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SET_RE = re.compile(r"MPLIB 2 - Set (\d+)$")
NUMBER_FIELDS = {
    "J", "I", "K", "a_I", "SP", "a_SP", "MP", "MF", "CR", "RD",
    "PD", "RS", "a_RS", "IS", "a_IS", "UF", "varUF",
}


def number(value: str) -> int | float | str:
    value = value.strip().replace(",", ".")
    if not value:
        return ""
    try:
        parsed = float(value)
    except ValueError:
        return value
    return int(parsed) if parsed.is_integer() else parsed


def set_directories(root: Path, selected: int | None) -> list[tuple[int, Path]]:
    result = []
    for directory in sorted(root.glob("MPLIB 2 - Set *")):
        match = SET_RE.fullmatch(directory.name)
        if match and directory.is_dir() and (selected is None or int(match.group(1)) == selected):
            result.append((int(match.group(1)), directory))
    if not result:
        raise FileNotFoundError(f"No MPLIB2 set directory found under {root}")
    return result


def read_summary(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if not reader.fieldnames:
            raise ValueError(f"empty CSV header: {path}")
        fields = [field.strip() for field in reader.fieldnames]
        rows = []
        for raw in reader:
            row = {field: number(raw.get(field, "")) for field in fields}
            rows.append(row)
    return fields, rows


def rcmp_header(path: Path) -> tuple[int, int]:
    """Read only the first two non-empty lines of an RCMP instance."""
    with path.open(encoding="utf-8", errors="replace") as stream:
        nonempty = (line.split() for line in stream if line.split())
        project_count = int(next(nonempty)[0])
        resource_count = int(next(nonempty)[0])
        return project_count, resource_count


def summarize_set(set_number: int, directory: Path) -> dict[str, Any]:
    summary_path = directory / f"Summary_Set_{set_number}.csv"
    fields, rows = read_summary(summary_path)
    # macOS archives may include AppleDouble metadata files named ._*.rcmp.
    instances = sorted(
        path
        for path in (directory / "Instances").glob("*.rcmp")
        if not path.name.startswith("._")
    )
    dimensions = Counter((row["J"], row["I"], row["K"]) for row in rows)
    headers = Counter(rcmp_header(path) for path in instances)

    metrics = {}
    for field in fields:
        if field not in NUMBER_FIELDS:
            continue
        values = [row[field] for row in rows if isinstance(row[field], (int, float))]
        if not values:
            continue
        metrics[field] = {
            "mean": round(sum(values) / len(values), 6),
            "min": min(values),
            "max": max(values),
            "unique": len(set(values)),
        }

    return {
        "set": set_number,
        "summary_rows": len(rows),
        "rcmp_files": len(instances),
        "counts_match": len(rows) == len(instances),
        "dimensions": [
            {"J": j, "I": i, "K": k, "count": count}
            for (j, i, k), count in sorted(dimensions.items())
        ],
        "rcmp_headers": [
            {"projects": projects, "resources": resources, "files": count}
            for (projects, resources), count in sorted(headers.items())
        ],
        "metrics": metrics,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [f"MPLIB2: {report['root']}", f"总实例数: {report['total_instances']}", ""]
    for item in report["sets"]:
        lines.append(f"Set {item['set']}: CSV={item['summary_rows']}, RCMP={item['rcmp_files']}, 数量一致={item['counts_match']}")
        dimensions = ", ".join(
            f"J={d['J']},I={d['I']},K={d['K']} ({d['count']})" for d in item["dimensions"]
        )
        lines.append(f"  规模: {dimensions}")
        headers = ", ".join(
            f"{h['projects']}项目/{h['resources']}资源 ({h['files']})" for h in item["rcmp_headers"]
        )
        lines.append(f"  文件头: {headers}")
        for field in ("SP", "CR", "RD", "PD", "RS", "IS", "UF", "varUF"):
            metric = item["metrics"].get(field)
            if metric:
                lines.append(f"  {field}: mean={metric['mean']}, min={metric['min']}, max={metric['max']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MPLIB2 CSV and RCMP files")
    parser.add_argument("root", nargs="?", type=Path, default=Path("data/MPLIB2"))
    parser.add_argument("--set", dest="set_number", type=int, help="summarize only one set")
    parser.add_argument("--json", type=Path, metavar="PATH", help="also write the complete JSON report")
    parser.add_argument("--output", type=Path, metavar="PATH", help="write the text report to a file")
    args = parser.parse_args()

    try:
        sets = [summarize_set(number_, directory) for number_, directory in set_directories(args.root, args.set_number)]
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report = {"root": str(args.root), "total_instances": sum(item["rcmp_files"] for item in sets), "sets": sets}
    text = format_report(report)
    print(text, end="")
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
