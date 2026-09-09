# tests/test_diag_quality.py — bisect the "makespan ~2x too high" anomaly
import sys, glob
import numpy as np

sys.path.insert(0, "src")
from rcpsp_data import load_instance
from rcpsp_env import RCPSPEnv
from baselines import run_policy, _tails

# ---------- A. raw-vs-parsed dump for j301_1 (eyeball vs the file) ----------
inst = load_instance("data/psplib/j30/j301_1.sm")
print("=== A. j301_1 parsed data ===")
print("durations :", inst.durations.tolist())
print("capacities:", inst.capacities.tolist())
print("demands   :")
print(inst.demands)
# compare visually with:  sed -n '1,60p' data/psplib/j30/j301_1.sm


# ---------- B. LFT on j301_1, record the order, cross-check with independent SGS ----------
def naive_sgs(inst, order):
    """Independent resource-aware serial SGS, written from scratch here."""
    start = np.zeros(inst.n_activities, dtype=int)
    usage = np.zeros(
        (int(inst.durations.sum()) + inst.n_activities + 2, inst.n_renewable), dtype=int
    )
    for j in order:
        d, need = inst.durations[j], inst.demands[j]
        t = 0
        for p in inst.predecessors[j]:
            t = max(t, start[p] + inst.durations[p])
        while (usage[t : t + d].sum(axis=0) + need > inst.capacities).any():
            t += 1
        start[j] = t
        usage[t : t + d] += need
    return int((start + inst.durations).max())


print("\n=== B. LFT on j301_1 (BKS = 43; expect 43-50) ===")
env = RCPSPEnv(inst)
env.reset()
order = []
while True:
    a = int(np.random.default_rng(1).choice(env.eligible))  # placeholder, replaced below
    # force LFT choice:
    from baselines import RULES, STATIC

    static = {"tails": _tails(inst)}
    score = -static["tails"]
    cand = env.eligible
    a = int(cand[np.argmin(score[cand])])
    _, _, term, _, info = env.step(a)
    order.append(a)
    if term:
        break
print("env makespan      :", info["makespan"])
print("naive SGS (same order):", naive_sgs(inst, order))
print("order:", order)

# ---------- C. topological_order validity over the whole corpus ----------
print("\n=== C. topo-order violations over corpus ===")
files = sorted(glob.glob("data/psplib/**/*.sm", recursive=True)) + sorted(
    glob.glob("data/oras/**/*.rcp", recursive=True)
)
bad = 0
for f in files:
    i = load_instance(f)
    pos = np.empty(i.n_activities, dtype=int)
    for k, j in enumerate(i.topological_order()):
        pos[j] = k
    for j in range(i.n_activities):
        for s in i.successors[j]:
            if pos[j] >= pos[s]:
                bad += 1
print("violations:", bad)

# ---------- D. LFT makespans, first 10 instances, individually ----------
print("\n=== D. LFT per-instance makespans ===")
for f in sorted(glob.glob("data/psplib/j30/*.sm"))[:10]:
    print(f.split("/")[-1], "->", run_policy(RCPSPEnv(load_instance(f)), "LFT"))
