# tests/test_baselines.py — priority rules vs random on j30 (BKS-known set)
import sys, glob
import numpy as np

sys.path.insert(0, "src")
from rcpsp_data import load_instance
from rcpsp_env import RCPSPEnv
from baselines import run_policy

np.random.seed(0)
files = sorted(glob.glob("data/psplib/j30/*.sm"))[:100]
print(f"{len(files)} j30 instances")

results = {}
for rule in ["random", "SPT", "GRPW", "MTS", "LFT"]:
    mk = []
    for f in files:
        env = RCPSPEnv(load_instance(f))
        mk.append(run_policy(env, rule))
        env.validate()
    results[rule] = np.array(mk)
    print(f"  {rule:6s}: mean={np.mean(mk):6.1f}  min={mk and min(mk):3d}  max={max(mk):3d}")

print(
    f"\nLFT beats random by "
    f"{results['random'].mean() - results['LFT'].mean():.1f} makespan units"
)
print("BASELINE TESTS PASSED")
