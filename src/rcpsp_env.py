"""
RCPSP as an MDP:
  state    : set of scheduled activities + their start times + resource usage profile
  action   : pick one activity from the eligible set (all predecessors scheduled)
  transit  : schedule it at its earliest resource-feasible start (serial SGS)
  terminal : all activities scheduled (incl. dummy sink)
  reward   : sparse, -makespan at the end (normalized)

Action space is Discrete(n) with an explicit eligibility mask (env.eligible,
also exposed via info["mask"]) -- perfect for MaskablePPO later.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rcpsp_data import RCPSPInstance


class RCPSPEnv(gym.Env):
    """Single-instance episode: schedule every activity once."""

    def __init__(self, instance: RCPSPInstance, reward_mode: str = "sparse"):
        super().__init__()
        assert reward_mode in ("sparse", "progress")
        self.inst = instance
        self.reward_mode = reward_mode

        self.n = instance.n_activities
        self.R = instance.n_renewable
        self.T = int(instance.durations.sum()) + 1  # safe makespan upper bound

        self.action_space = spaces.Discrete(self.n)
        # per-activity: [dur_norm, scheduled, eligible, start_norm] + R demand feats
        self.obs_dim = self.n * (4 + self.R) + self.R
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.reset()

    # ---------------- core ----------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.scheduled = np.zeros(self.n, dtype=bool)
        self.start = np.zeros(self.n, dtype=np.int64)
        self.finish = np.zeros(self.n, dtype=np.int64)
        # cumulative resource usage: cum[u, r] = usage of resource r in [0, u)
        self.usage = np.zeros((self.T + 1, self.R), dtype=np.int64)

        self._update_eligible()
        return self._obs(), {}

    def _update_eligible(self):
        preds_done = np.array(
            [all(self.scheduled[p] for p in self.inst.predecessors[j]) for j in range(self.n)]
        )
        self.eligible = np.where(preds_done & ~self.scheduled)[0]

    def _earliest_feasible(self, j: int) -> int:
        """Earliest t >= EST(j) such that resources allow [t, t+dur)."""
        d = self.inst.durations[j]
        est = int(self.finish[self.inst.predecessors[j]].max()) if self.inst.predecessors[j] else 0
        need = self.inst.demands[j]  # (R,)
        cap = self.inst.capacities  # (R,)
        for t in range(est, self.T - d + 1):
            used = self.usage[t : t + d].sum(axis=0)  # window sum, (R,)
            if (used + need <= cap).all():
                return t
        raise RuntimeError(f"no feasible start for activity {j} within horizon")

    def step(self, action: int):
        action = int(action)
        if action not in set(self.eligible.tolist()):
            return self._obs(), 0.0, False, False, {"error": "ineligible action"}

        # schedule at earliest feasible time
        t = self._earliest_feasible(action)
        d = self.inst.durations[action]
        self.start[action], self.finish[action] = t, t + d
        self.usage[t:t + d] += self.inst.demands[action]

        self.scheduled[action] = True
        self._update_eligible()

        terminated = bool(self.scheduled.all())
        if terminated:
            makespan = int(self.finish.max())
            reward = -makespan / self.T  # sparse, normalized
            info = {"makespan": makespan}
        else:
            reward = 0.0
            info = {}
            if self.reward_mode == "progress":  # dense shaping
                reward = -d / self.T
        info["mask"] = self._mask()
        return self._obs(), reward, terminated, False, info

    # ---------------- helpers for RL ----------------

    def _mask(self) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        m[self.eligible] = True
        return m

    def _obs(self) -> np.ndarray:
        i = self.inst
        max_dur = max(i.durations.max(), 1)
        act_feats = np.concatenate(
            [
                (i.durations / max_dur).reshape(-1, 1),  # normalized duration
                self.scheduled.reshape(-1, 1).astype(np.float32),
                self._mask().reshape(-1, 1).astype(np.float32),
                (self.start / self.T).reshape(-1, 1),  # normalized start
                (i.demands / i.capacities).astype(np.float32),  # per-resource load
            ],
            axis=1,
        )
        # current resource usage profile summary: utilization now
        now = int(self.finish.max()) if self.scheduled.any() else 0
        util = (self.usage[now] / i.capacities).astype(np.float32)
        return np.concatenate([act_feats.ravel(), util]).astype(np.float32)

    def validate(self) -> bool:
        """Post-episode check: precedence + resource feasibility."""
        for j in range(self.n):
            for p in self.inst.predecessors[j]:
                assert self.finish[p] <= self.start[j], f"prec violation {p}->{j}"
        for r in range(self.R):
            assert (
                self.usage[:, r] <= self.inst.capacities[r]
            ).all(), f"capacity violation on resource {r}"
        return True
