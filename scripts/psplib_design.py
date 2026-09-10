"""Read the nominal PSPLIB factor design from the RCPLIB workbook.

``data/bks/RCPLIB (Parameters and BKS).xlsx`` (sheet ``All``) carries one row
per benchmark instance with the realised generator parameters in the columns
``CNC`` (network complexity), ``RF`` (resource factor) and ``RS`` (resource
strength, Kolisch's 0-1 capacity-slack definition -- a different formula, and a
different scale, from the demand-spread RS this repo measures from the files).

This module turns those rows into the *nominal* factor grid PSPLIB was
generated from, snapping each realised value to its design level:

* ``NC``  in {1.5, 1.8, 2.1}          (realised ~1.4 / 1.73 / 2.07)
* ``RF``  in {0.25, 0.5, 0.75, 1.0}   (realised ~0.26 / 0.51 / 0.76)
* ``RS``  in {0.2, 0.5, 0.7, 0.9} for j30-j90 and {0.1, ..., 0.5} for j120

The result is authoritative (no clustering heuristics) and is what
``scripts/generate_pool.py --mode psp-grid`` uses to lay out the grid cells.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

NOMINAL_NC = (1.5, 1.8, 2.1)
NOMINAL_RF = (0.25, 0.5, 0.75, 1.0)
# j30-j90 use four RS levels, j120 five (and lower ones -- bigger projects are
# tighter under the same nominal RS).
NOMINAL_RS_BY_SIZE = {
    30: (0.2, 0.5, 0.7, 0.9),
    60: (0.2, 0.5, 0.7, 0.9),
    90: (0.2, 0.5, 0.7, 0.9),
    120: (0.1, 0.2, 0.3, 0.4, 0.5),
}

_SET_RE = re.compile(r"j(30|60|90|120)(\d+)_")


@dataclass(frozen=True)
class PsplibSet:
    """One PSPLIB set (10 instances) with its nominal design coordinates."""

    size: int
    set_id: int
    nc: float
    rf: float
    rs: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.size, self.set_id)


def _nearest(value: float, levels: tuple[float, ...]) -> float:
    return min(levels, key=lambda level: abs(level - value))


def load_psplib_design(xlsx_path: Path) -> list[PsplibSet]:
    """Parse the workbook and return one nominal coordinate per PSPLIB set."""
    import openpyxl

    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet = workbook["All"]
    rows = sheet.iter_rows(values_only=True)
    header = [str(h) for h in next(rows)]
    col = {h: i for i, h in enumerate(header)}
    for required in ("SetName", "FileName", "CNC", "RF", "RS"):
        if required not in col:
            raise ValueError(f"RCPLIB workbook sheet 'All' lacks column {required!r}")

    realised: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for row in rows:
        if str(row[col["SetName"]]).strip().lower() != "psplib":
            continue
        match = _SET_RE.match(str(row[col["FileName"]]))
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        realised.setdefault(key, []).append(
            (
                float(row[col["CNC"]]),
                float(row[col["RF"]]),
                float(row[col["RS"]]),
            )
        )

    design: list[PsplibSet] = []
    for (size, set_id), values in sorted(realised.items()):
        cnc = sum(v[0] for v in values) / len(values)
        rf = sum(v[1] for v in values) / len(values)
        rs = sum(v[2] for v in values) / len(values)
        design.append(
            PsplibSet(
                size=size,
                set_id=set_id,
                nc=_nearest(cnc, NOMINAL_NC),
                rf=_nearest(rf, NOMINAL_RF),
                rs=_nearest(rs, NOMINAL_RS_BY_SIZE[size]),
            )
        )
    if not design:
        raise ValueError(f"no PSPLIB rows found in {xlsx_path}")
    return design
