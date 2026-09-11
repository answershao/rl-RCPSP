"""Gymnasium environment for activity-selection RCPSP scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.rcpsp import (
    ActivityId,
    Instance,
    Schedule,
    preview_serial_sgs_insert,
    serial_sgs_insert,
    validate_schedule,
)
from src.core.rules import rule_makespan
from src.envs.observation import (
    DYNAMIC_ACTIVITY_FEATURE_COUNT,
    RESOURCE_PROFILE_BIN_COUNT,
    RESOURCE_PROFILE_CHANNEL_COUNT,
    RESOURCE_PROFILE_FEATURE_COUNT,
)


# A masked actor never emits an illegal action, so this path is a safety net
# rather than a training signal.  It used to return -1.0, which is larger than
# a *whole* episode return (about -1 to -2 once the reward is scaled by
# ``time_scale``): if the mask ever failed (NaN logits, fp16 overflow, a
# padding bug) a single rejected action would dominate the value target.
# Zero keeps the net harmless; ``info["invalid_action"]`` still reports it and
# the step budget below bounds the episode.
INVALID_ACTION_PENALTY = 0.0

# Static priority rule whose single-pass serial SGS makespan defines the
# per-instance ``time_scale`` (see ``RCPSPEnv.time_scale``).
TIME_SCALE_RULE = "LST"

# A well-behaved episode schedules exactly one activity per step, so it ends
# after ``activity_count`` steps.  Sustained rejected actions would otherwise
# loop forever (nothing else bounds an episode); the budget truncates instead.
STEP_BUDGET_FACTOR = 2


@dataclass
class _ScheduleState:
    """Mutable schedule-construction state for one environment episode."""

    starts: dict[ActivityId, int]
    finishes: dict[ActivityId, int]
    usage: np.ndarray
    remaining_predecessors: np.ndarray
    eligible_mask: np.ndarray
    current_time: int = 0
    critical_lower_bound: int = 0
    resource_work: int = 0
    invalid_action_penalty: float = 0.0
    steps: int = 0
    terminated: bool = False
    aborted: bool = False

    @property
    def over(self) -> bool:
        """True once the episode is finished for either reason."""
        return self.terminated or self.aborted

    @classmethod
    def create(cls, activity_count: int, resource_count: int, horizon: int) -> _ScheduleState:
        return cls(
            starts={},
            finishes={},
            usage=np.zeros((horizon + 1, resource_count), dtype=np.int32),
            remaining_predecessors=np.zeros(activity_count, dtype=np.int32),
            eligible_mask=np.zeros(activity_count, dtype=bool),
        )

    def reset(self, predecessor_counts: np.ndarray) -> None:
        self.starts.clear()
        self.finishes.clear()
        self.usage.fill(0)
        self.remaining_predecessors[:] = predecessor_counts
        self.eligible_mask[:] = self.remaining_predecessors == 0
        self.current_time = 0
        self.critical_lower_bound = 0
        self.resource_work = 0
        self.invalid_action_penalty = 0.0
        self.steps = 0
        self.terminated = False
        self.aborted = False


class RCPSPEnv(gym.Env[dict[str, np.ndarray], int]):
    """Construct a serial SSGS schedule by selecting eligible activities.

    One step selects one precedence-eligible activity and inserts it at its
    earliest resource-feasible time. The episode therefore has exactly one
    scheduling decision per activity.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance: Instance,
        *,
        reward_shaping_coef: float = 0.0,
    ):
        super().__init__()
        if reward_shaping_coef < 0.0:
            raise ValueError("reward_shaping_coef must be non-negative")
        self.reward_shaping_coef = float(reward_shaping_coef)
        self.instance = instance
        self.activity_ids = tuple(sorted(self.instance.activities))
        self.activity_index = {activity_id: i for i, activity_id in enumerate(self.activity_ids)}
        self.activity_count = len(self.activity_ids)
        self.resource_count = self.instance.resource_count
        self.horizon = sum(activity.duration for activity in self.instance.activities.values())
        # ``time_scale`` is the normalisation reference for every time-like
        # feature (the six dynamic activity features, the current-time global
        # feature and the resource-profile bins).  ``horizon`` -- the duration
        # sum -- is a valid upper bound but it is 2.3-5.3x larger than any
        # makespan a serial SGS actually produces, so normalising by it threw
        # away most of each feature's range: measured on j120,
        # ``makespan_increment`` only ever used 0.016 of [0, 1] and 124 of the
        # 128 profile channels were identically zero.  One LST-priority serial
        # SGS pass (~1 ms per instance, once) gives a size-consistent and
        # policy-relevant reference instead.
        # NOTE this is a *reference*, not an upper bound: a poor policy can
        # exceed it, so every consumer clips to [0, 1] (see ``_observation``
        # and ``flatten_observation``) instead of silently emitting values > 1.
        self.time_scale = max(
            int(rule_makespan(self.instance, TIME_SCALE_RULE, scheme="serial")), 1
        )
        self._step_budget = STEP_BUDGET_FACTOR * self.activity_count

        self.action_space = spaces.Discrete(self.activity_count)
        int_max = np.iinfo(np.int32).max
        capacities = np.asarray(self.instance.capacities, dtype=np.int32)
        max_duration = max((activity.duration for activity in self.instance.activities.values()), default=0)
        self.observation_space = spaces.Dict(
            {
                "activity_status": spaces.Box(0, 2, (self.activity_count,), dtype=np.int8),
                "precedence_satisfied": spaces.MultiBinary(self.activity_count),
                "durations": spaces.Box(0, max(max_duration, 1), (self.activity_count,), dtype=np.int32),
                "resource_demands": spaces.Box(
                    0, int_max, (self.activity_count, self.resource_count), dtype=np.int32
                ),
                "remaining_capacity": spaces.Box(
                    np.zeros(self.resource_count, dtype=np.int32), capacities, dtype=np.int32
                ),
                "resource_profile": spaces.Box(
                    0.0,
                    1.0,
                    (self.resource_count, RESOURCE_PROFILE_FEATURE_COUNT),
                    dtype=np.float32,
                ),
                "current_time": spaces.Box(0, max(self.horizon, 1), (1,), dtype=np.int32),
                "eligible_mask": spaces.MultiBinary(self.activity_count),
                "dynamic_activity_features": spaces.Box(
                    0.0,
                    1.0,
                    (self.activity_count, DYNAMIC_ACTIVITY_FEATURE_COUNT),
                    dtype=np.float32,
                ),
            }
        )

        self._durations = np.asarray(
            [self.instance.activities[item].duration for item in self.activity_ids], dtype=np.int32
        )
        self._demands = np.asarray(
            [self.instance.activities[item].demand for item in self.activity_ids], dtype=np.int32
        )
        self._resource_work_by_activity = self._durations * self._demands.sum(axis=1)
        self._capacities = capacities
        self._capacity_limits = self._capacities - self._demands
        self._capacity_scale = np.maximum(self._capacities.astype(np.float32), 1.0)
        self._capacity_total = int(np.sum(capacities))
        self._predecessor_counts = np.asarray(
            [len(self.instance.predecessors[activity_id]) for activity_id in self.activity_ids],
            dtype=np.int32,
        )
        downstream_cache: dict[ActivityId, int] = {}

        def downstream_duration(activity_id: ActivityId) -> int:
            if activity_id not in downstream_cache:
                activity = self.instance.activities[activity_id]
                downstream_cache[activity_id] = activity.duration + max(
                    (downstream_duration(item) for item in activity.successors),
                    default=0,
                )
            return downstream_cache[activity_id]

        self._downstream_durations = np.asarray(
            [downstream_duration(item) for item in self.activity_ids], dtype=np.int32
        )
        # Denominator for the ``makespan_increment`` feature.  The increment an
        # insertion can cost is bounded by the largest single activity, so
        # dividing by max(d) puts the feature on a suite-independent [0, 1]
        # scale.  Dividing by ``time_scale`` (the other five features) left its
        # ceiling at 0.09-0.20 depending on the suite -- a 2x distribution
        # drift the actor head had to absorb -- and dividing by the candidate's
        # *own* duration collapses it to {0, 1}, because an undelayed insertion
        # costs exactly its duration (measured: 0 or 1.00, nothing between).
        self._max_duration = max(int(self._durations.max(initial=0)), 1)
        # The horizon is a safe upper bound for every serial schedule. Keeping
        # usage in an array avoids Python list growth and nested update loops.
        self._state = _ScheduleState.create(
            self.activity_count, self.resource_count, self.horizon
        )
        self._status = np.zeros(self.activity_count, dtype=np.int8)
        self._precedence = np.zeros(self.activity_count, dtype=np.int8)
        self._terminal_eligible_mask = np.zeros(self.activity_count, dtype=np.int8)
        self._remaining_capacity = np.zeros(self.resource_count, dtype=np.int32)
        self._resource_profile = np.zeros(
            (self.resource_count, RESOURCE_PROFILE_FEATURE_COUNT),
            dtype=np.float32,
        )
        self._resource_profile_sums = np.zeros(
            (self.resource_count, RESOURCE_PROFILE_BIN_COUNT), dtype=np.int64
        )
        self._resource_profile_ranges = tuple(
            self._profile_range(bin_index)
            for bin_index in range(RESOURCE_PROFILE_BIN_COUNT)
        )
        self._finish_times = np.full(self.activity_count, -1, dtype=np.int32)
        self._current_time = np.zeros(1, dtype=np.int32)
        self._dynamic_activity_features = np.zeros(
            (self.activity_count, DYNAMIC_ACTIVITY_FEATURE_COUNT), dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._state.reset(self._predecessor_counts)
        self._resource_profile.fill(0.0)
        self._resource_profile_sums.fill(0)
        self._finish_times.fill(-1)
        observation = self._observation()
        return observation, {"eligible_mask": observation["eligible_mask"].copy()}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        state = self._state
        if state.over:
            raise RuntimeError("step() called after the episode finished; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.activity_count})")
        requested_index = int(action)
        if not state.eligible_mask.any():
            raise RuntimeError("precedence graph is cyclic or has a missing predecessor")
        if not state.eligible_mask[requested_index]:
            return self.reject_action()
        state.steps += 1
        chosen_index = requested_index
        chosen = self.activity_ids[chosen_index]

        old_time = state.current_time
        old_critical_lower_bound = state.critical_lower_bound
        start, finish = serial_sgs_insert(
            self.instance, chosen, state.starts, state.finishes, state.usage,
            capacities_array=self._capacities,
            demand_array=self._demands[chosen_index],
            capacity_limit_array=self._capacity_limits[chosen_index],
            current_makespan=state.current_time,
        )
        state.resource_work += int(self._resource_work_by_activity[chosen_index])
        self._finish_times[chosen_index] = finish
        self._update_resource_profile(start, finish, self._demands[chosen_index])
        state.eligible_mask[chosen_index] = False
        for successor in self.instance.activities[chosen].successors:
            successor_index = self.activity_index[successor]
            state.remaining_predecessors[successor_index] -= 1
            if state.remaining_predecessors[successor_index] == 0:
                state.eligible_mask[successor_index] = True
        state.current_time = max(state.current_time, finish)
        remaining_path_duration = max(
            0,
            int(self._downstream_durations[chosen_index])
            - int(self._durations[chosen_index]),
        )
        state.critical_lower_bound = max(
            old_critical_lower_bound,
            state.current_time,
            finish + remaining_path_duration,
        )
        # Dividing by ``time_scale`` (not the duration sum) makes the episode
        # return exactly ``-makespan / time_scale`` at gamma=1, and pins the
        # value target near O(1) instead of the ~0.15 a /horizon denominator
        # produced -- which left the critic gradient two orders of magnitude
        # weaker than the actor's.
        scale = float(max(self.time_scale, 1))
        makespan_penalty = -float(state.current_time - old_time) / scale
        critical_path_penalty = -float(
            state.critical_lower_bound - old_critical_lower_bound
        ) / scale
        reward = makespan_penalty + self.reward_shaping_coef * critical_path_penalty
        state.terminated = len(state.starts) == self.activity_count

        observation = self._observation()
        info: dict[str, Any] = {
            "chosen_activity": chosen,
            "start": start,
            "finish": finish,
            "makespan": state.current_time,
            "eligible_mask": observation["eligible_mask"].copy(),
            "invalid_action": False,
        }
        if state.terminated:
            info["schedule"] = self.schedule
        return self._finalize(observation, reward, info)

    def reject_action(
        self,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance an episode that received an action the mask should have blocked.

        Also used by the padded multi-instance wrapper, whose action space is
        the global activity cap and can therefore address indices beyond this
        instance's ``activity_count``.  The step still counts against the budget
        so sustained rejection truncates the episode instead of looping.
        """
        state = self._state
        if state.over:
            raise RuntimeError("step() called after the episode finished; call reset()")
        state.steps += 1
        state.invalid_action_penalty += INVALID_ACTION_PENALTY
        observation = self._observation()
        return self._finalize(
            observation,
            INVALID_ACTION_PENALTY,
            {
                "makespan": state.current_time,
                "eligible_mask": observation["eligible_mask"].copy(),
                "invalid_action": True,
            },
        )

    def _finalize(
        self,
        observation: dict[str, np.ndarray],
        reward: float,
        info: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Attach episode metrics and decide whether to truncate.

        A healthy episode schedules one activity per step and terminates after
        ``activity_count`` steps.  The budget only bites when the action mask
        failed and the policy keeps proposing illegal actions: the episode is
        truncated (not terminated) instead of looping forever.
        """
        state = self._state
        truncated = False
        if not state.terminated and state.steps >= self._step_budget:
            state.aborted = True
            truncated = True
            info["aborted"] = True
        if state.terminated or truncated:
            scale = float(max(self.time_scale, 1))
            # Monitor persists ``TERMINAL_METRICS`` by direct dict lookup, so
            # these keys must exist on truncated episodes as well.  ``schedule``
            # stays termination-only: an aborted episode has no complete
            # schedule to validate.
            info.update(
                {
                    "normalized_makespan": state.current_time / scale,
                    "resource_utilization": self._resource_utilization(state.current_time),
                    "activity_count": self.activity_count,
                    "episode_makespan_penalty": -state.current_time / scale,
                    "episode_critical_path_penalty": (
                        -state.critical_lower_bound / scale
                    ),
                    "episode_invalid_action_penalty": state.invalid_action_penalty,
                    "episode_reward": (
                        -state.current_time / scale
                        - self.reward_shaping_coef * state.critical_lower_bound / scale
                        + state.invalid_action_penalty
                    ),
                }
            )
        return observation, reward, state.terminated, truncated, info

    @property
    def schedule(self) -> Schedule:
        """Return and validate the completed schedule."""
        state = self._state
        if not state.terminated:
            raise RuntimeError("the schedule is only available after episode termination")
        schedule = Schedule(dict(state.starts), dict(state.finishes), state.current_time)
        validate_schedule(self.instance, schedule)
        return schedule

    def _resource_utilization(self, end_time: int) -> float:
        """Return aggregate capacity utilization over the current frontier."""
        if end_time <= 0:
            return 0.0
        capacity = float(end_time * self._capacity_total)
        used = float(self._state.resource_work)
        return used / capacity if capacity > 0 else 0.0

    def _profile_range(self, bin_index: int) -> tuple[int, int]:
        """Bin edges over ``[0, time_scale)``, final bin widened to ``horizon``.

        Spreading the bins over the reference horizon instead of the duration
        sum puts all of them where activities are actually scheduled.  The
        final bin is widened to the full usage horizon because ``time_scale``
        is only a reference: a policy can overrun it, and those insertions must
        not silently fall outside the profile.
        """
        span = max(self.time_scale, 1)
        start = (bin_index * span) // RESOURCE_PROFILE_BIN_COUNT
        stop = ((bin_index + 1) * span) // RESOURCE_PROFILE_BIN_COUNT
        if bin_index == RESOURCE_PROFILE_BIN_COUNT - 1:
            stop = max(stop, self.horizon)
        # Degenerate spans (very short instances) would otherwise produce empty
        # bins; collapse them onto the first available slot.
        if start >= stop and self.horizon > 0:
            start = min(start, self.horizon - 1)
            stop = start + 1
        return start, stop

    def _update_resource_profile(
        self, start: int, finish: int, demand: np.ndarray
    ) -> None:
        """Refresh only profile bins touched by the latest SGS insertion."""
        if start == finish or self.horizon == 0:
            return
        profile = self._resource_profile.reshape(
            self.resource_count,
            RESOURCE_PROFILE_CHANNEL_COUNT,
            RESOURCE_PROFILE_BIN_COUNT,
        )
        for bin_index, (bin_start, bin_stop) in enumerate(
            self._resource_profile_ranges
        ):
            overlap = max(0, min(finish, bin_stop) - max(start, bin_start))
            if not overlap:
                continue
            self._resource_profile_sums[:, bin_index] += demand * overlap
            width = bin_stop - bin_start
            profile[:, 0, bin_index] = (
                self._resource_profile_sums[:, bin_index]
                / width
                / self._capacity_scale
            )
            profile[:, 1, bin_index] = (
                self._state.usage[bin_start:bin_stop].max(axis=0)
                / self._capacity_scale
            )

    def _observation(self) -> dict[str, np.ndarray]:
        status = self._status
        status.fill(0)
        state = self._state
        scheduled = self._finish_times >= 0
        status[scheduled] = np.where(
            state.terminated | (self._finish_times[scheduled] < state.current_time),
            2,
            1,
        )

        precedence = self._precedence
        np.equal(state.remaining_predecessors, 0, out=precedence)
        eligible_mask = state.eligible_mask if not state.terminated else self._terminal_eligible_mask
        # The latest occupied interval gives a useful capacity signal at the
        # partial schedule frontier; at time zero no resource is occupied.
        frontier_usage = state.usage[state.current_time - 1] if state.current_time > 0 else 0
        remaining = self._remaining_capacity
        np.subtract(self._capacities, frontier_usage, out=remaining)
        self._current_time[0] = state.current_time
        dynamic = self._dynamic_activity_features
        dynamic.fill(0.0)
        scale = float(max(self.time_scale, 1))
        for activity_index in np.flatnonzero(eligible_mask):
            activity_id = self.activity_ids[int(activity_index)]
            ready = max(
                (state.finishes[pred] for pred in self.instance.predecessors[activity_id]),
                default=0,
            )
            start, finish = preview_serial_sgs_insert(
                self.instance,
                activity_id,
                state.finishes,
                state.usage,
                capacities_array=self._capacities,
                demand_array=self._demands[activity_index],
                capacity_limit_array=self._capacity_limits[activity_index],
                current_makespan=state.current_time,
            )
            # Clipped because a bad policy can schedule past the reference
            # horizon; the observation space declares [0, 1].  Note the fifth
            # feature is scaled by the largest activity, not by ``time_scale``
            # -- see ``self._max_duration``.
            dynamic[activity_index] = (
                min(ready / scale, 1.0),
                min(start / scale, 1.0),
                min(finish / scale, 1.0),
                min(max(start - ready, 0) / scale, 1.0),
                min(max(finish - state.current_time, 0) / self._max_duration, 1.0),
                min((start + self._downstream_durations[activity_index]) / scale, 1.0),
            )
        return {
            "activity_status": status,
            "precedence_satisfied": precedence,
            # These arrays are immutable instance data; avoid allocating copies
            # on every environment step.  Flattening converts them to float32.
            "durations": self._durations,
            "resource_demands": self._demands,
            "remaining_capacity": remaining,
            "resource_profile": self._resource_profile,
            "current_time": self._current_time,
            "eligible_mask": eligible_mask,
            "dynamic_activity_features": dynamic,
        }
