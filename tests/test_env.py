# tests/test_env.py — 100 RG30 episodes with a random policy
import sys, glob, random
import numpy as np

sys.path.insert(0, "src")
from rcpsp_data import load_instance
from rcpsp_env import RCPSPEnv

random.seed(42)
files = sorted(glob.glob("data/oras/RCPSP/RG30/**/*.rcp", recursive=True))[:100]
print(f"testing on {len(files)} RG30 instances")

mk = []
for k, f in enumerate(files):
    env = RCPSPEnv(load_instance(f))
    obs, _ = env.reset()
    steps = 0
    while True:
        a = random.choice(env.eligible.tolist())  # random valid action
        obs, r, term, trunc, info = env.step(a)
        steps += 1
        assert steps <= env.n  # no infinite episodes
        if term:
            break
    env.validate()  # precedence + capacity
    mk.append(info["makespan"])
    if (k + 1) % 25 == 0:
        print(f"  {k+1}: mean={np.mean(mk):.1f}")

mk = np.array(mk)
print(
    f"\nmakespan over {len(mk)} episodes: "
    f"mean={mk.mean():.1f}  min={mk.min()}  max={mk.max()}  std={mk.std():.1f}"
)
print("ENV TESTS PASSED")
