"""Priority-rule policies on top of RCPSPEnv (serial SGS by priority)."""

import numpy as np


def _tails(inst):
    """tail[j] = min possible delay from end of j to project end (critical path)."""
    tail = inst.durations.copy()
    for j in reversed(inst.topological_order()):
        for s in inst.successors[j]:
            tail[j] = max(tail[j], inst.durations[j] + tail[s])
    return tail


def _n_transitive_succ(inst):
    n = inst.n_activities
    reach = [set(inst.successors[j]) for j in range(n)]
    for j in reversed(range(n)):
        for s in list(inst.successors[j]):
            reach[j] |= reach[s]
    return np.array([len(r) for r in reach])


RULES = {
    "random": lambda env, i: np.random.rand(env.n),
    "SPT": lambda env, i: env.inst.durations,  # shortest proc time
    "LFT": lambda env, i: -i["tails"],  # latest finish time
    "MTS": lambda env, i: -i["nsucc"],  # most total successors
    "GRPW": lambda env, i: -(env.inst.durations + i["nsucc"]),  # rank positional weight
}

STATIC = {"tails": _tails, "nsucc": _n_transitive_succ}


def run_policy(env, rule: str) -> int:
    static = {
        k: fn(env.inst)
        for k, fn in STATIC.items()
        if any(k in r.__code__.co_names or k in str(r.__code__.co_consts) for r in [RULES[rule]])
    }
    while True:
        score = RULES[rule](env, static)
        cand = env.eligible
        a = cand[np.argmin(score[cand])]
        _, _, term, _, info = env.step(int(a))
        if term:
            return info["makespan"]
