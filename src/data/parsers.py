"""
Unified parser for RCPSP instance files.
Formats:
  - .sm  : PSPLIB / Patterson format (keyword sections)
  - .rcp : OR&S / ProGen format (raw numbers)
Both produce RCPSPInstance with:
  - 0-indexed activities, activity 0 = dummy source, n-1 = dummy sink
  - successors / predecessors as adjacency lists
  - durations[act], demands[act, res], capacities[res]
"""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class RCPSPInstance:
    name: str
    n_activities: int  # including dummy source & sink
    n_renewable: int
    horizon: int
    durations: np.ndarray  # (n,) int, durations[i] >= 0 (0 for dummies)
    demands: np.ndarray  # (n, R) int
    capacities: np.ndarray  # (R,) int
    successors: list[list[int]]  # 0-indexed, adjacency lists
    predecessors: list[list[int]] = field(init=False)

    def __post_init__(self):
        self.predecessors = [[] for _ in range(self.n_activities)]
        for i, succs in enumerate(self.successors):
            for s in succs:
                self.predecessors[s].append(i)

    def topological_order(self) -> list[int]:
        """Kahn's algorithm — valid serial-SGS order; raises if the graph cycles."""
        indeg = [len(p) for p in self.predecessors]
        order, queue = [], [i for i in range(self.n_activities) if indeg[i] == 0]
        while queue:
            i = queue.pop()
            order.append(i)
            for s in self.successors[i]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    queue.append(s)
        assert len(order) == self.n_activities, "cycle detected!"
        return order


# ---------------- .sm (PSPLIB) parser ----------------


def parse_sm(path) -> RCPSPInstance:
    text = Path(path).read_text()
    tokens = text.split()
    lines = text.splitlines()

    def first_int_after(keyword):
        """First int strictly after token `keyword` in the token stream."""
        i = tokens.index(keyword) + 1
        while True:
            try:
                return int(tokens[i])
            except ValueError:
                i += 1

    # --- header ---
    n_jobs = first_int_after("jobs")
    horizon = first_int_after("horizon")
    n_renew = first_int_after("renewable")

    # --- precedence: token-based (rows may wrap across lines) ---
    p = tokens.index("RELATIONS:") + 1
    successors = []
    for expected in range(1, n_jobs + 1):

        def read_int(p):  # skip non-numeric tokens
            while True:
                try:
                    return int(tokens[p]), p + 1
                except ValueError:
                    p += 1

        job, p = read_int(p)
        assert job == expected, f"precedence misaligned: got job {job}, want {expected}"
        n_modes, p = read_int(p)
        assert n_modes == 1, "only single-mode RCPSP supported"
        n_succ, p = read_int(p)
        succs = []
        for _ in range(n_succ):
            s, p = read_int(p)
            succs.append(s - 1)  # 0-indexed
        successors.append(succs)

    # --- requests/durations: LINE-based, skip header rows containing digits ---
    start = next(i for i, ln in enumerate(lines) if "REQUESTS/DURATIONS" in ln)
    durations = np.zeros(n_jobs, dtype=np.int64)
    demands = np.zeros((n_jobs, n_renew), dtype=np.int64)
    expected = 1
    for ln in lines[start + 1 :]:
        parts = ln.split()
        # accept only rows whose first token is the expected jobnr
        if not parts or not parts[0].isdigit() or int(parts[0]) != expected:
            continue  # skips "jobnr. mode duration R 1..." & "----"
        assert len(parts) >= 3 + n_renew, f"short request row: {ln!r}"
        job, mode = int(parts[0]), int(parts[1])
        durations[job - 1] = int(parts[2])
        for r in range(n_renew):
            demands[job - 1, r] = int(parts[3 + r])
        expected += 1
        if expected > n_jobs:
            break
    assert expected == n_jobs + 1, f"request rows incomplete, stopped at job {expected - 1}"

    # --- capacities: first all-integer LINE after RESOURCEAVAILABILITIES ---
    capacities = None
    seen = False
    for ln in lines:
        if "RESOURCEAVAILABILITIES" in ln:
            seen = True
            continue
        if seen:
            parts = ln.split()
            if len(parts) >= n_renew and all(x.lstrip("-").isdigit() for x in parts):
                capacities = np.array([int(x) for x in parts[:n_renew]], dtype=np.int64)
                break
    assert capacities is not None, "capacities row not found"

    return RCPSPInstance(
        name=Path(path).stem,
        n_activities=n_jobs,
        n_renewable=n_renew,
        horizon=horizon,
        durations=durations,
        demands=demands,
        capacities=capacities,
        successors=successors,
    )


# ---------------- .rcp (OR&S / ProGen) parser ----------------


def parse_rcp(path) -> RCPSPInstance:
    tokens = Path(path).read_text().split()
    pos = 0

    def next_int():
        nonlocal pos
        pos += 1
        return int(tokens[pos - 1])

    n_jobs = next_int()
    n_renew = next_int()
    capacities = np.array([next_int() for _ in range(n_renew)], dtype=np.int64)

    durations = np.zeros(n_jobs, dtype=np.int64)
    demands = np.zeros((n_jobs, n_renew), dtype=np.int64)
    successors = []
    for j in range(n_jobs):
        durations[j] = next_int()
        for r in range(n_renew):
            demands[j, r] = next_int()
        n_succ = next_int()
        successors.append([next_int() - 1 for _ in range(n_succ)])  # -> 0-indexed

    return RCPSPInstance(
        name=Path(path).stem,
        n_activities=n_jobs,
        n_renewable=n_renew,
        horizon=int(durations.sum()),  # safe upper bound (no per-file horizon in .rcp)
        durations=durations,
        demands=demands,
        capacities=capacities,
        successors=successors,
    )


def load_instance(path: str | Path) -> RCPSPInstance:
    path = Path(path)
    return parse_sm(path) if path.suffix == ".sm" else parse_rcp(path)
