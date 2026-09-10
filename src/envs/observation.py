"""Dynamic observations and static graph caches for RCPSP policies.

Observations target single-project RCPSP instances (0-indexed activity ids)
produced by ``src.data.adapter``; padded multi-instance environments keep
the shapes fixed across the j30-j120 / RG300 activity range.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from src.core.rcpsp import Instance


# Upper bound over per-activity successor counts across the whole single-project
# corpus.  PSPLIB j30-j120 fan out to at most 24, RG30 reaches 25 and the large
# RG300 generalization set reaches 89, so the policy-side successor tensor needs
# this headroom to keep one model valid across every suite.
MAX_SUCCESSORS = 96
# The same scan puts the worst in-degree at 91 (RG300), so both directions share
# the 96 headroom and the two degree features stay on a comparable scale.
MAX_PREDECESSORS = 96
RESOURCE_PROFILE_BIN_COUNT = 16
RESOURCE_PROFILE_CHANNEL_COUNT = 2
RESOURCE_PROFILE_FEATURE_COUNT = (
    RESOURCE_PROFILE_BIN_COUNT * RESOURCE_PROFILE_CHANNEL_COUNT
)
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
    """Named slices for the compact, dynamic observation contract."""

    max_activities: int
    max_resources: int

    def __post_init__(self) -> None:
        if min(self.max_activities, self.max_resources) < 1:
            raise ValueError("observation dimensions must be positive")

    @property
    def activity_status(self) -> slice:
        return slice(0, self.max_activities)

    @property
    def precedence_satisfied(self) -> slice:
        return slice(self.activity_status.stop, self.activity_status.stop + self.max_activities)

    @property
    def eligible_mask(self) -> slice:
        return slice(
            self.precedence_satisfied.stop,
            self.precedence_satisfied.stop + self.max_activities,
        )

    @property
    def dynamic_activity_features(self) -> slice:
        return slice(
            self.eligible_mask.stop,
            self.eligible_mask.stop
            + self.max_activities * DYNAMIC_ACTIVITY_FEATURE_COUNT,
        )

    @property
    def remaining_capacity(self) -> slice:
        return slice(
            self.dynamic_activity_features.stop,
            self.dynamic_activity_features.stop + self.max_resources,
        )

    @property
    def resource_profile(self) -> slice:
        return slice(
            self.remaining_capacity.stop,
            self.remaining_capacity.stop
            + self.max_resources * RESOURCE_PROFILE_FEATURE_COUNT,
        )

    @property
    def current_time(self) -> int:
        return self.resource_profile.stop

    @property
    def global_features(self) -> slice:
        return slice(self.remaining_capacity.start, self.current_time + 1)

    @property
    def instance_index(self) -> int:
        return self.current_time + 1

    @property
    def size(self) -> int:
        return self.instance_index + 1


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

        activity_mask[instance_index, :activity_count] = 1.0
        for node_index, activity_id in enumerate(activity_ids):
            activity = instance.activities[activity_id]
            if len(activity.successors) > max_successors:
                raise ValueError(f"activity {activity_id} has too many successors")
            predecessor_count = len(instance.predecessors.get(activity_id, ()))
            if predecessor_count > MAX_PREDECESSORS:
                raise ValueError(f"activity {activity_id} has too many predecessors")
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
    )


def flatten_observation(
    observation: Mapping[str, np.ndarray],
    capacities: tuple[int, ...],
    horizon: int,
    *,
    instance_index: int = 0,
    catalog_size: int = 1,
    max_activities: int | None = None,
    max_resources: int | None = None,
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

    layout = ObservationLayout(max_activities, max_resources)
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
        max_resources,
        RESOURCE_PROFILE_CHANNEL_COUNT,
        RESOURCE_PROFILE_BIN_COUNT,
    )
    np.copyto(
        profile[:resource_count],
        observation["resource_profile"].reshape(
            resource_count,
            RESOURCE_PROFILE_CHANNEL_COUNT,
            RESOURCE_PROFILE_BIN_COUNT,
        ),
        casting="unsafe",
    )
    result[layout.current_time] = observation["current_time"][0] / max(horizon, 1)
    # Zero is reserved for malformed/padded observations.
    result[layout.instance_index] = (instance_index + 1) / catalog_size
    return result


def observation_size(activity_count: int, resource_count: int) -> int:
    """Return the compact dynamic observation length."""
    return ObservationLayout(activity_count, resource_count).size
