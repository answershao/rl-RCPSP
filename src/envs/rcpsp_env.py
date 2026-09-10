"""Gymnasium environment for activity-selection RCPSP scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
from src.data.adapter import load_core_instance
from src.envs.observation import (
    DYNAMIC_ACTIVITY_FEATURE_COUNT,
    RESOURCE_PROFILE_BIN_COUNT,
    RESOURCE_PROFILE_CHANNEL_COUNT,
    RESOURCE_PROFILE_FEATURE_COUNT,
)


INVALID_ACTION_PENALTY = -1.0


@dataclass
class _ScheduleState:
    """Mutable schedule-construction state for one environment episode."""

    starts: dict[ActivityId, int]
    finishes: dict[ActivityId, int]
    usage: np.ndarray
    remaining_predecessors: np.ndarray
    eligible_mask: np.ndarray
    current_time: int = 0
    resource_work: int = 0
    invalid_action_penalty: float = 0.0
    terminated: bool = False

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
        self.resource_work = 0
        self.invalid_action_penalty = 0.0
        self.terminated = False


class RCPSPEnv(gym.Env[dict[str, np.ndarray], int]):
    """Construct a serial SSGS schedule by selecting eligible activities.

    One step selects one precedence-eligible activity and inserts it at its
    earliest resource-feasible time. The episode therefore has exactly one
    scheduling decision per activity.
    """

    metadata = {"render_modes": []}

    def __init__(self, instance: Instance | str | Path):
        super().__init__()
        self.instance = (
            load_core_instance(instance) if isinstance(instance, (str, Path)) else instance
        )
        self.activity_ids = tuple(sorted(self.instance.activities))
        self.activity_index = {activity_id: i for i, activity_id in enumerate(self.activity_ids)}
        self.activity_count = len(self.activity_ids)
        self.resource_count = self.instance.resource_count
        self.horizon = sum(activity.duration for activity in self.instance.activities.values())

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
        if state.terminated:
            raise RuntimeError("step() called after the episode terminated; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.activity_count})")
        requested_index = int(action)
        if not state.eligible_mask.any():
            raise RuntimeError("precedence graph is cyclic or has a missing predecessor")
        invalid_action = not state.eligible_mask[requested_index]
        if invalid_action:
            state.invalid_action_penalty += INVALID_ACTION_PENALTY
            observation = self._observation()
            return (
                observation,
                INVALID_ACTION_PENALTY,
                False,
                False,
                {
                    "makespan": state.current_time,
                    "eligible_mask": observation["eligible_mask"].copy(),
                    "invalid_action": True,
                },
            )
        chosen_index = requested_index
        chosen = self.activity_ids[chosen_index]

        old_time = state.current_time
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
        makespan_penalty = -float(state.current_time - old_time) / max(self.horizon, 1)
        reward = makespan_penalty
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
            # Keep terminal metrics scalar so Monitor and vectorized SB3
            # environments can persist them in their episode records.
            info.update(
                {
                    "normalized_makespan": state.current_time / max(self.horizon, 1),
                    "resource_utilization": self._resource_utilization(state.current_time),
                    "activity_count": self.activity_count,
                    "episode_makespan_penalty": -state.current_time / max(self.horizon, 1),
                    "episode_invalid_action_penalty": state.invalid_action_penalty,
                    "episode_reward": (
                        -state.current_time / max(self.horizon, 1)
                        + state.invalid_action_penalty
                    ),
                }
            )
        return observation, reward, state.terminated, False, info

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
        start = (bin_index * self.horizon) // RESOURCE_PROFILE_BIN_COUNT
        stop = ((bin_index + 1) * self.horizon) // RESOURCE_PROFILE_BIN_COUNT
        if start == stop and self.horizon > 0:
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
        horizon = max(self.horizon, 1)
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
            dynamic[activity_index] = (
                ready / horizon,
                start / horizon,
                finish / horizon,
                (start - ready) / horizon,
                max(0, finish - state.current_time) / horizon,
                (start + self._downstream_durations[activity_index]) / horizon,
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
