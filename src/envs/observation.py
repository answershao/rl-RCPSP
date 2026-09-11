"""Dynamic observations and static graph caches for RCPSP policies.

Observations target single-project RCPSP instances (0-indexed activity ids)
produced by ``src.data.adapter``; padded multi-instance environments keep
shapes fixed across the j30-j120 activity range.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np

from src.core.rcpsp import Instance, latest_start_times


# Full-corpus bounds over the generated train/validation pool and held-out
# PSPLIB j30-j120 suites are 19 successors and 18 predecessors. One spare slot
# keeps the cache contract explicit while avoiding oversized edge tensors.
MAX_SUCCESSORS = 20
MAX_PREDECESSORS = 20
DYNAMIC_ACTIVITY_FEATURES = (
    "precedence_ready_time",
    "earliest_start",
    "earliest_finish",
    "resource_wait",
    "makespan_increment",
    "critical_completion",
)
DYNAMIC_ACTIVITY_FEATURE_COUNT = len(DYNAMIC_ACTIVITY_FEATURES)


@dataclass(frozen=True)
class ObservationLayout:
    """Named slices for the exact dynamic Markov-state observation."""

    max_activities: int
    max_resources: int
    max_horizon: int

    def __post_init__(self) -> None:
        if min(self.max_activities, self.max_resources) < 1:
            raise ValueError("observation dimensions must be positive")
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be positive")

    @cached_property
    def activity_status(self) -> slice:
        return slice(0, self.max_activities)

    @cached_property
    def precedence_satisfied(self) -> slice:
        return slice(self.activity_status.stop, self.activity_status.stop + self.max_activities)

    @cached_property
    def eligible_mask(self) -> slice:
        return slice(
            self.precedence_satisfied.stop,
            self.precedence_satisfied.stop + self.max_activities,
        )

    @cached_property
    def remaining_predecessors(self) -> slice:
        return slice(
            self.eligible_mask.stop,
            self.eligible_mask.stop + self.max_activities,
        )

    @cached_property
    def scheduled_start_times(self) -> slice:
        return slice(
            self.remaining_predecessors.stop,
            self.remaining_predecessors.stop + self.max_activities,
        )

    @cached_property
    def scheduled_finish_times(self) -> slice:
        return slice(
            self.scheduled_start_times.stop,
            self.scheduled_start_times.stop + self.max_activities,
        )

    @cached_property
    def dynamic_activity_features(self) -> slice:
        return slice(
            self.scheduled_finish_times.stop,
            self.scheduled_finish_times.stop
            + self.max_activities * DYNAMIC_ACTIVITY_FEATURE_COUNT,
        )

    @cached_property
    def remaining_capacity(self) -> slice:
        return slice(
            self.dynamic_activity_features.stop,
            self.dynamic_activity_features.stop + self.max_resources,
        )

    @cached_property
    def resource_profile(self) -> slice:
        return slice(
            self.remaining_capacity.stop,
            self.remaining_capacity.stop
            + self.max_resources * self.max_horizon,
        )

    @cached_property
    def current_time(self) -> int:
        return self.resource_profile.stop

    @cached_property
    def global_features(self) -> slice:
        return slice(self.remaining_capacity.start, self.time_scale + 1)

    @cached_property
    def critical_lower_bound(self) -> int:
        return self.current_time + 1

    @cached_property
    def resource_work(self) -> int:
        return self.critical_lower_bound + 1

    @cached_property
    def steps(self) -> int:
        return self.resource_work + 1

    @cached_property
    def invalid_action_penalty(self) -> int:
        return self.steps + 1

    @cached_property
    def terminated(self) -> int:
        return self.invalid_action_penalty + 1

    @cached_property
    def aborted(self) -> int:
        return self.terminated + 1

    @cached_property
    def horizon(self) -> int:
        return self.aborted + 1

    @cached_property
    def time_scale(self) -> int:
        return self.horizon + 1

    @cached_property
    def instance_index(self) -> int:
        return self.time_scale + 1

    @cached_property
    def size(self) -> int:
        return self.instance_index + 1


@lru_cache(maxsize=None)
def observation_layout(
    max_activities: int,
    max_resources: int,
    max_horizon: int,
) -> ObservationLayout:
    """Reuse immutable slice metadata on the per-environment-step hot path."""
    return ObservationLayout(
        max_activities,
        max_resources,
        max_horizon=max_horizon,
    )


@dataclass(frozen=True)
class StaticGraphCache:
    """Normalized immutable graph data indexed by catalog instance."""

    instance_names: tuple[str, ...]
    durations: np.ndarray
    resource_demands: np.ndarray
    successor_indices: np.ndarray
    successor_counts: np.ndarray
    predecessor_counts: np.ndarray
    downstream_durations: np.ndarray
    activity_mask: np.ndarray
    slack_ratios: np.ndarray
    on_critical_path: np.ndarray

    @property
    def instance_count(self) -> int:
        return int(self.durations.shape[0])


def build_static_graph_cache(
    instances: Sequence[Instance],
    *,
    max_activities: int,
    max_resources: int,
    max_successors: int = MAX_SUCCESSORS,
) -> StaticGraphCache:
    """Build policy-side tensors for data that never changes during an episode."""
    if not instances:
        raise ValueError("instances must not be empty")
    instance_count = len(instances)
    instance_names = tuple(instance.name for instance in instances)
    if len(set(instance_names)) != instance_count:
        raise ValueError("static cache instance names must be unique")
    durations = np.zeros((instance_count, max_activities), dtype=np.float32)
    resource_demands = np.zeros(
        (instance_count, max_activities, max_resources), dtype=np.float32
    )
    successor_indices = np.full(
        (instance_count, max_activities, max_successors), -1, dtype=np.int64
    )
    successor_counts = np.zeros((instance_count, max_activities), dtype=np.float32)
    predecessor_counts = np.zeros((instance_count, max_activities), dtype=np.float32)
    downstream_durations = np.zeros((instance_count, max_activities), dtype=np.float32)
    activity_mask = np.zeros((instance_count, max_activities), dtype=np.float32)
    slack_ratios = np.zeros((instance_count, max_activities), dtype=np.float32)
    on_critical_path = np.zeros((instance_count, max_activities), dtype=np.float32)

    for instance_index, instance in enumerate(instances):
        activity_ids = tuple(sorted(instance.activities))
        activity_count = len(activity_ids)
        resource_count = instance.resource_count
        if activity_count > max_activities or resource_count > max_resources:
            raise ValueError("static cache dimensions are smaller than an instance")
        activity_positions = {
            activity_id: index for index, activity_id in enumerate(activity_ids)
        }
        # Duration features are normalised by per-instance *maxima*, not by the
        # duration sum.  A sum-based denominator grows with n (horizon = sum(d)
        # ~ n * E[d]), so duration/horizon ~ 1/n drifts out of the training
        # distribution as instance size leaves the training pool -- measured
        # feature overlap with PSPLIB dropped 0.97 (j30) -> 0.22 (j120) because
        # of exactly this.  Max-based denominators are size invariant and keep
        # every feature inside [0, 1].
        duration_scale = max(
            (item.duration for item in instance.activities.values()), default=1
        )
        capacity_scale = np.maximum(np.asarray(instance.capacities, dtype=np.float32), 1.0)
        longest_paths: dict[int, int] = {}

        def downstream_duration(activity_id: int) -> int:
            if activity_id not in longest_paths:
                activity = instance.activities[activity_id]
                longest_paths[activity_id] = activity.duration + max(
                    (downstream_duration(item) for item in activity.successors),
                    default=0,
                )
            return longest_paths[activity_id]

        # Memoised full traversal; the per-node loop below then hits the cache.
        downstream_scale = max(
            (
                downstream_duration(activity_id)
                for activity_id in activity_ids
                if instance.activities[activity_id].duration > 0
            ),
            default=1,
        )

        # CPM slack, evaluated against a *critical-path* deadline instead of the
        # duration sum.  With T = CP every slack is in [0, CP], so slack / CP is
        # scale free across j30-j120, and slack == 0 marks exactly the critical
        # path -- the single most informative structural signal in RCPSP (LST
        # underlies MSLK, WCS and most GPHH terminals).  Normalising by sum(d)
        # would reintroduce the n-dependent drift that the duration features
        # already had to be fixed for.
        earliest_starts: dict[int, int] = {}

        def earliest_start(activity_id: int) -> int:
            if activity_id not in earliest_starts:
                earliest_starts[activity_id] = max(
                    (
                        earliest_start(predecessor)
                        + instance.activities[predecessor].duration
                        for predecessor in instance.predecessors.get(activity_id, ())
                    ),
                    default=0,
                )
            return earliest_starts[activity_id]

        critical_path = max(
            (
                earliest_start(activity_id) + downstream_duration(activity_id)
                for activity_id in activity_ids
            ),
            default=0,
        )
        latest_starts = latest_start_times(instance, horizon=critical_path)
        slack_scale = max(critical_path, 1)

        activity_mask[instance_index, :activity_count] = 1.0
        for node_index, activity_id in enumerate(activity_ids):
            activity = instance.activities[activity_id]
            if len(activity.successors) > max_successors:
                raise ValueError(f"activity {activity_id} has too many successors")
            predecessor_count = len(instance.predecessors.get(activity_id, ()))
            if predecessor_count > MAX_PREDECESSORS:
                raise ValueError(f"activity {activity_id} has too many predecessors")
            slack = max(latest_starts[activity_id] - earliest_start(activity_id), 0)
            slack_ratios[instance_index, node_index] = slack / slack_scale
            on_critical_path[instance_index, node_index] = float(slack == 0)
            durations[instance_index, node_index] = activity.duration / max(
                duration_scale, 1
            )
            resource_demands[
                instance_index, node_index, :resource_count
            ] = np.asarray(activity.demand, dtype=np.float32) / capacity_scale
            successor_counts[instance_index, node_index] = (
                len(activity.successors) / max_successors
            )
            predecessor_counts[instance_index, node_index] = (
                predecessor_count / MAX_PREDECESSORS
            )
            downstream_durations[instance_index, node_index] = (
                downstream_duration(activity_id) / downstream_scale
            )
            for slot, successor in enumerate(activity.successors):
                successor_indices[instance_index, node_index, slot] = (
                    activity_positions[successor]
                )

    return StaticGraphCache(
        instance_names=instance_names,
        durations=durations,
        resource_demands=resource_demands,
        successor_indices=successor_indices,
        successor_counts=successor_counts,
        predecessor_counts=predecessor_counts,
        downstream_durations=downstream_durations,
        activity_mask=activity_mask,
        slack_ratios=slack_ratios,
        on_critical_path=on_critical_path,
    )


def flatten_observation(
    observation: Mapping[str, np.ndarray],
    capacities: tuple[int, ...],
    *,
    instance_index: int = 0,
    catalog_size: int = 1,
    max_activities: int | None = None,
    max_resources: int | None = None,
    max_horizon: int,
    capacity_scale: np.ndarray | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Flatten only scheduling state that changes between decisions."""
    activity_count = observation["activity_status"].size
    resource_count = len(capacities)
    max_activities = max_activities or activity_count
    max_resources = max_resources or resource_count
    if max_activities < activity_count or max_resources < resource_count:
        raise ValueError("padding dimensions cannot be smaller than the observation")
    if catalog_size < 1 or not 0 <= instance_index < catalog_size:
        raise ValueError("instance_index must identify an entry in the static catalog")

    required = {
        "remaining_predecessors",
        "scheduled_start_times",
        "scheduled_finish_times",
        "critical_lower_bound",
        "resource_work",
        "steps",
        "invalid_action_penalty",
        "terminated",
        "aborted",
        "horizon",
        "time_scale",
    }
    missing = sorted(required.difference(observation))
    if missing:
        raise ValueError(f"exact observation is missing fields: {missing}")
    local_horizon = int(observation["resource_profile"].shape[1])
    if max_horizon < local_horizon:
        raise ValueError("max_horizon cannot be smaller than the observation horizon")
    layout = observation_layout(max_activities, max_resources, max_horizon)
    if out is None:
        result = np.zeros(layout.size, dtype=np.float32)
    else:
        if out.shape != (layout.size,) or out.dtype != np.float32:
            raise ValueError("out must be a float32 array with the flattened observation shape")
        result = out
        result.fill(0.0)

    np.multiply(
        observation["activity_status"],
        0.5,
        out=result[layout.activity_status.start:layout.activity_status.start + activity_count],
        casting="unsafe",
    )
    np.copyto(
        result[
            layout.precedence_satisfied.start:
            layout.precedence_satisfied.start + activity_count
        ],
        observation["precedence_satisfied"],
        casting="unsafe",
    )
    np.copyto(
        result[layout.eligible_mask.start:layout.eligible_mask.start + activity_count],
        observation["eligible_mask"],
        casting="unsafe",
    )
    np.divide(
        observation["remaining_predecessors"],
        float(MAX_PREDECESSORS),
        out=result[
            layout.remaining_predecessors.start:
            layout.remaining_predecessors.start + activity_count
        ],
        casting="unsafe",
    )
    horizon_scale = float(max_horizon)
    np.divide(
        np.maximum(observation["scheduled_start_times"], 0),
        horizon_scale,
        out=result[
            layout.scheduled_start_times.start:
            layout.scheduled_start_times.start + activity_count
        ],
        casting="unsafe",
    )
    np.divide(
        np.maximum(observation["scheduled_finish_times"], 0),
        horizon_scale,
        out=result[
            layout.scheduled_finish_times.start:
            layout.scheduled_finish_times.start + activity_count
        ],
        casting="unsafe",
    )
    dynamic = result[layout.dynamic_activity_features].reshape(
        max_activities, DYNAMIC_ACTIVITY_FEATURE_COUNT
    )
    np.copyto(
        dynamic[:activity_count],
        observation["dynamic_activity_features"],
        casting="unsafe",
    )
    capacities_array = (
        capacity_scale
        if capacity_scale is not None
        else np.maximum(np.asarray(capacities, dtype=np.float32), 1.0)
    )
    np.divide(
        observation["remaining_capacity"],
        capacities_array,
        out=result[
            layout.remaining_capacity.start:
            layout.remaining_capacity.start + resource_count
        ],
        casting="unsafe",
    )
    profile = result[layout.resource_profile].reshape(
        max_resources, max_horizon
    )
    profile[:resource_count, :local_horizon] = observation["resource_profile"]
    result[layout.current_time] = observation["current_time"][0] / horizon_scale
    result[layout.critical_lower_bound] = (
        observation["critical_lower_bound"][0] / horizon_scale
    )
    result[layout.resource_work] = observation["resource_work"][0] / max(
        float(np.sum(capacities) * max_horizon), 1.0
    )
    result[layout.steps] = observation["steps"][0] / max(
        float(2 * activity_count), 1.0
    )
    result[layout.invalid_action_penalty] = observation["invalid_action_penalty"][0]
    result[layout.terminated] = observation["terminated"][0]
    result[layout.aborted] = observation["aborted"][0]
    result[layout.horizon] = observation["horizon"][0] / horizon_scale
    result[layout.time_scale] = observation["time_scale"][0] / horizon_scale
    # Zero is reserved for malformed/padded observations.
    result[layout.instance_index] = (instance_index + 1) / catalog_size
    return result


def observation_size(
    activity_count: int,
    resource_count: int,
    max_horizon: int,
) -> int:
    """Return the compact dynamic observation length."""
    return observation_layout(
        activity_count,
        resource_count,
        max_horizon,
    ).size
