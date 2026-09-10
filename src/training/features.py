"""Shared directed-GIN encoder and independent actor/critic heads."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from src.envs.observation import (
    DYNAMIC_ACTIVITY_FEATURE_COUNT,
    ObservationLayout,
    StaticGraphCache,
)


class DirectedGINLayer(nn.Module):
    """GIN update with separate predecessor and successor relations."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.epsilon = nn.Parameter(th.zeros(1))
        self.predecessor_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.successor_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.update = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        embeddings: th.Tensor,
        predecessor_message: th.Tensor,
        successor_message: th.Tensor,
    ) -> th.Tensor:
        aggregate = (
            (1.0 + self.epsilon) * embeddings
            + self.predecessor_projection(predecessor_message)
            + self.successor_projection(successor_message)
        )
        return self.update(aggregate)


def _edge_degrees(
    edge_sources: th.Tensor,
    edge_targets: th.Tensor,
    edge_mask: th.Tensor,
    node_count: int,
    dtype: th.dtype,
) -> tuple[th.Tensor, th.Tensor]:
    """Exact in-degree and out-degree per node, counted from the compact edges.

    Padding edges carry ``edge_mask == False``, so they contribute zero weight.
    """
    weight = edge_mask.to(dtype)
    shape = (edge_mask.shape[0], node_count)
    in_degree = th.zeros(shape, dtype=dtype, device=edge_mask.device)
    in_degree.scatter_add_(1, edge_targets, weight)
    out_degree = th.zeros(shape, dtype=dtype, device=edge_mask.device)
    out_degree.scatter_add_(1, edge_sources, weight)
    return in_degree, out_degree


def _aggregate_edge_messages(
    embeddings: th.Tensor,
    edge_sources: th.Tensor,
    edge_targets: th.Tensor,
    edge_mask: th.Tensor,
    in_degree: th.Tensor,
    out_degree: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """Mean-pool messages in both directions using only compact, real edges.

    Summing would scale each node's message with its degree while the encoder
    ends in ReLU, so nothing cancels. Dividing by the degree
    keeps message magnitudes comparable across suites; the raw counts stay
    available to the network as explicit node features.
    """
    embedding_dim = embeddings.shape[-1]
    expanded_sources = edge_sources.unsqueeze(-1).expand(-1, -1, embedding_dim)
    expanded_targets = edge_targets.unsqueeze(-1).expand(-1, -1, embedding_dim)
    valid_edges = edge_mask.unsqueeze(-1).to(embeddings.dtype)
    source_messages = embeddings.gather(1, expanded_sources) * valid_edges
    target_messages = embeddings.gather(1, expanded_targets) * valid_edges

    predecessor_sum = th.zeros_like(embeddings)
    predecessor_sum.scatter_add_(1, expanded_targets, source_messages)
    successor_sum = th.zeros_like(embeddings)
    successor_sum.scatter_add_(1, expanded_sources, target_messages)
    predecessor_message = predecessor_sum / in_degree.clamp(min=1.0).unsqueeze(-1)
    successor_message = successor_sum / out_degree.clamp(min=1.0).unsqueeze(-1)
    return predecessor_message, successor_message


def _compact_edge_indices(
    edge_sources: th.Tensor,
    edge_targets: th.Tensor,
    edge_mask: th.Tensor,
    node_count: int,
) -> tuple[th.Tensor, th.Tensor]:
    """Flatten a padded edge batch into indices for one disjoint graph."""
    batch_offsets = (
        th.arange(edge_sources.shape[0], device=edge_sources.device).unsqueeze(1)
        * node_count
    )
    return (
        (edge_sources + batch_offsets)[edge_mask],
        (edge_targets + batch_offsets)[edge_mask],
    )


def _aggregate_compact_edge_messages(
    embeddings: th.Tensor,
    flat_sources: th.Tensor,
    flat_targets: th.Tensor,
    in_degree: th.Tensor,
    out_degree: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """Aggregate a batch without materializing its padded edge slots."""
    flat_embeddings = embeddings.flatten(0, 1)
    predecessor_sum = th.zeros_like(flat_embeddings)
    predecessor_sum.index_add_(0, flat_targets, flat_embeddings[flat_sources])
    successor_sum = th.zeros_like(flat_embeddings)
    successor_sum.index_add_(0, flat_sources, flat_embeddings[flat_targets])
    predecessor_message = predecessor_sum.view_as(embeddings)
    successor_message = successor_sum.view_as(embeddings)
    predecessor_message = predecessor_message / in_degree.clamp(min=1.0).unsqueeze(-1)
    successor_message = successor_message / out_degree.clamp(min=1.0).unsqueeze(-1)
    return predecessor_message, successor_message


class SharedDirectedGINExtractor(BaseFeaturesExtractor):
    """Encode an RCPSP graph once for both policy and value estimation."""

    def __init__(
        self,
        observation_space,
        *,
        max_activities: int,
        max_resources: int,
        static_cache: StaticGraphCache,
        max_successors: int = 3,
        gin_layers: int = 3,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        global_dim: int = 16,
        mixed_precision: str = "none",
    ) -> None:
        if gin_layers < 1:
            raise ValueError("gin_layers must be positive")
        self.layout = ObservationLayout(max_activities, max_resources)
        if observation_space.shape != (self.layout.size,):
            raise ValueError(
                f"expected flattened RCPSP observation {(self.layout.size,)}, "
                f"got {observation_space.shape}"
            )
        features_dim = max_activities * embedding_dim + global_dim + 2 * max_activities
        super().__init__(observation_space, features_dim)
        self.max_activities = max_activities
        self.max_resources = max_resources
        self.max_successors = max_successors
        self.embedding_dim = embedding_dim
        self.global_dim = global_dim
        self.mixed_precision = mixed_precision
        self.set_static_cache(static_cache)

        global_input_dim = self.layout.global_features.stop - self.layout.global_features.start
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, global_dim),
            nn.ReLU(),
        )
        activity_input_dim = (
            7 + DYNAMIC_ACTIVITY_FEATURE_COUNT + max_resources + global_dim
        )
        self.activity_encoder = nn.Sequential(
            nn.Linear(activity_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.gin_layers = nn.ModuleList(
            DirectedGINLayer(embedding_dim, hidden_dim) for _ in range(gin_layers)
        )

    def set_static_cache(self, static_cache: StaticGraphCache) -> None:
        """Replace immutable instance data, for example before a new evaluation split."""
        self.instance_count = static_cache.instance_count
        self.instance_names = static_cache.instance_names
        expected_node_shape = (self.instance_count, self.max_activities)
        if static_cache.durations.shape != expected_node_shape:
            raise ValueError("static cache does not match max_activities")
        if static_cache.resource_demands.shape != (
            self.instance_count, self.max_activities, self.max_resources
        ):
            raise ValueError("static cache does not match max_resources")
        if static_cache.successor_indices.shape != (
            self.instance_count, self.max_activities, self.max_successors
        ):
            raise ValueError("static cache does not match max_successors")
        arrays = {
            "static_durations": static_cache.durations,
            "static_resource_demands": static_cache.resource_demands,
            "static_successor_indices": static_cache.successor_indices,
            "static_successor_counts": static_cache.successor_counts,
            "static_predecessor_counts": static_cache.predecessor_counts,
            "static_downstream_durations": static_cache.downstream_durations,
            "static_activity_mask": static_cache.activity_mask,
        }
        device = self.static_durations.device if hasattr(self, "static_durations") else None
        for name, array in arrays.items():
            tensor = th.from_numpy(array).to(device=device)
            if name in self._buffers:
                setattr(self, name, tensor)
            else:
                self.register_buffer(name, tensor)

        # Keep successor_indices above for checkpoint compatibility, but use a
        # compact edge list during message passing. Padding now scales with the
        # largest real edge count instead of max_activities * max_successors.
        valid_edges = static_cache.successor_indices >= 0
        edge_counts = valid_edges.sum(axis=(1, 2))
        max_edges = int(edge_counts.max(initial=0))
        edge_sources = np.zeros((self.instance_count, max_edges), dtype=np.int64)
        edge_targets = np.zeros((self.instance_count, max_edges), dtype=np.int64)
        edge_mask = np.zeros((self.instance_count, max_edges), dtype=np.bool_)
        in_degrees = np.zeros(expected_node_shape, dtype=np.float32)
        out_degrees = np.zeros(expected_node_shape, dtype=np.float32)
        source_slots = np.broadcast_to(
            np.arange(self.max_activities, dtype=np.int64)[:, None],
            static_cache.successor_indices.shape[1:],
        )
        for instance_index, edge_count in enumerate(edge_counts):
            count = int(edge_count)
            edge_sources[instance_index, :count] = source_slots[valid_edges[instance_index]]
            edge_targets[instance_index, :count] = static_cache.successor_indices[
                instance_index
            ][valid_edges[instance_index]]
            edge_mask[instance_index, :count] = True
            np.add.at(out_degrees[instance_index], edge_sources[instance_index, :count], 1)
            np.add.at(in_degrees[instance_index], edge_targets[instance_index, :count], 1)
        compact_arrays = {
            "static_edge_sources": edge_sources,
            "static_edge_targets": edge_targets,
            "static_edge_mask": edge_mask,
            "static_in_degrees": in_degrees,
            "static_out_degrees": out_degrees,
        }
        for name, array in compact_arrays.items():
            tensor = th.from_numpy(array).to(device=device)
            if name in self._buffers:
                setattr(self, name, tensor)
            else:
                self.register_buffer(name, tensor, persistent=False)

    def _autocast(self, tensor: th.Tensor):
        if not tensor.is_cuda or self.mixed_precision == "none":
            return nullcontext()
        dtype = th.bfloat16 if self.mixed_precision == "bf16" else th.float16
        return th.autocast(device_type="cuda", dtype=dtype)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        with self._autocast(observations):
            features = self._forward(observations)
        # Keep distributions and PPO losses in FP32; autocast is only needed
        # for the encoder's linear algebra.
        return features.float()

    def _forward(self, observations: th.Tensor) -> th.Tensor:
        n = self.max_activities
        layout = self.layout
        status = observations[:, layout.activity_status]
        precedence = observations[:, layout.precedence_satisfied]
        eligible = observations[:, layout.eligible_mask]
        dynamic = observations[:, layout.dynamic_activity_features].reshape(
            -1, n, DYNAMIC_ACTIVITY_FEATURE_COUNT
        )
        instance_indices = (
            th.round(observations[:, layout.instance_index] * self.instance_count).long() - 1
        ).clamp(min=0, max=self.instance_count - 1)
        durations = self.static_durations[instance_indices]
        demands = self.static_resource_demands[instance_indices]
        successor_counts = self.static_successor_counts[instance_indices]
        predecessor_counts = self.static_predecessor_counts[instance_indices]
        downstream_durations = self.static_downstream_durations[instance_indices]
        activity_mask = self.static_activity_mask[instance_indices]
        global_embedding = self.global_encoder(observations[:, layout.global_features])

        global_by_activity = global_embedding.unsqueeze(1).expand(-1, n, -1)
        activity_features = th.cat(
            [
                status.unsqueeze(-1),
                precedence.unsqueeze(-1),
                durations.unsqueeze(-1),
                eligible.unsqueeze(-1),
                successor_counts.unsqueeze(-1),
                predecessor_counts.unsqueeze(-1),
                downstream_durations.unsqueeze(-1),
                demands,
                dynamic,
                global_by_activity,
            ],
            dim=-1,
        )
        embeddings = self.activity_encoder(activity_features)
        valid_nodes = activity_mask.unsqueeze(-1)
        embeddings = embeddings * valid_nodes

        edge_sources = self.static_edge_sources[instance_indices]
        edge_targets = self.static_edge_targets[instance_indices]
        edge_mask = self.static_edge_mask[instance_indices]
        in_degree = self.static_in_degrees[instance_indices].to(embeddings.dtype)
        out_degree = self.static_out_degrees[instance_indices].to(embeddings.dtype)

        # CPU scatter/gather cost is dominated by the padded edge dimension
        # (313 maximum versus about 110 edges on average in the training pool).
        # Compact once and reuse the disjoint-graph indices for every GIN layer.
        # Keep the fixed-shape path for CUDA and torch.compile, where dynamic
        # boolean indexing can inhibit graph capture and padded kernels fare better.
        compact_edges = embeddings.device.type == "cpu" and not th.compiler.is_compiling()
        if compact_edges:
            flat_sources, flat_targets = _compact_edge_indices(
                edge_sources, edge_targets, edge_mask, n
            )

        for layer in self.gin_layers:
            if compact_edges:
                predecessor_message, successor_message = (
                    _aggregate_compact_edge_messages(
                        embeddings,
                        flat_sources,
                        flat_targets,
                        in_degree,
                        out_degree,
                    )
                )
            else:
                predecessor_message, successor_message = _aggregate_edge_messages(
                    embeddings,
                    edge_sources,
                    edge_targets,
                    edge_mask,
                    in_degree,
                    out_degree,
                )
            embeddings = (
                layer(embeddings, predecessor_message, successor_message) * valid_nodes
            )

        return th.cat(
            [
                embeddings.reshape(-1, n * self.embedding_dim),
                global_embedding,
                activity_mask,
                eligible,
            ],
            dim=1,
        )


def pool_graph_embedding(
    embeddings: th.Tensor,
    activity_mask: th.Tensor,
    global_embedding: th.Tensor,
) -> th.Tensor:
    """Summarize node embeddings with statistics that do not scale with node count.

    Both the mean and the max are bounded by the per-node embedding range, so
    the pooled vector stays in the same numeric range whether an instance has
    32 activities (j30) or 122 (j120). A node-count-dependent sum would grow
    linearly with the number of real activities and push the critic outside the
    value range it was trained on.

    Note this bounds the *node count* factor only. Message passing still has to
    normalize neighbour aggregation separately.
    """
    valid = activity_mask.unsqueeze(-1)
    masked = embeddings * valid
    node_count = valid.sum(dim=1)
    embedding_mean = masked.sum(dim=1) / node_count.clamp(min=1.0)
    sentinel = th.finfo(masked.dtype).min
    embedding_max = masked.masked_fill(valid <= 0.5, sentinel).max(dim=1).values
    # Every instance has at least one activity, but never let the sentinel leak
    # into the critic for an all-padding row.
    embedding_max = th.where(
        node_count > 0.5, embedding_max, th.zeros_like(embedding_max)
    )
    return th.cat([embedding_mean, embedding_max, global_embedding], dim=1)


class GINActorCriticHeads(nn.Module):
    """Independent node-scoring actor and graph-pooling critic."""

    def __init__(
        self,
        *,
        max_activities: int,
        embedding_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        mixed_precision: str = "none",
    ) -> None:
        super().__init__()
        self.max_activities = max_activities
        self.embedding_dim = embedding_dim
        self.global_dim = global_dim
        self.mixed_precision = mixed_precision
        self.latent_dim_pi = max_activities
        self.latent_dim_vf = 1
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            # Input is [node-mean(E) | node-max(E) | global(G)], see
            # pool_graph_embedding; both node statistics are size-stable.
            nn.Linear(2 * embedding_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _autocast(self, tensor: th.Tensor):
        if not tensor.is_cuda or self.mixed_precision == "none":
            return nullcontext()
        dtype = th.bfloat16 if self.mixed_precision == "bf16" else th.float16
        return th.autocast(device_type="cuda", dtype=dtype)

    def _unpack(
        self, features: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        n = self.max_activities
        embedding_stop = n * self.embedding_dim
        global_stop = embedding_stop + self.global_dim
        mask_stop = global_stop + n
        embeddings = features[:, :embedding_stop].reshape(-1, n, self.embedding_dim)
        global_embedding = features[:, embedding_stop:global_stop]
        activity_mask = features[:, global_stop:mask_stop]
        eligible = features[:, mask_stop:mask_stop + n]
        return embeddings, global_embedding, activity_mask, eligible

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        with self._autocast(features):
            embeddings, _, activity_mask, eligible = self._unpack(features)
            logits = self.actor(embeddings).squeeze(-1)
            legal = (activity_mask > 0.5) & (eligible > 0.5)
            logits = logits.masked_fill(~legal, th.finfo(logits.dtype).min)
        return logits.float()

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        with self._autocast(features):
            embeddings, global_embedding, activity_mask, _ = self._unpack(features)
            graph_embedding = pool_graph_embedding(
                embeddings, activity_mask, global_embedding
            )
            values = self.critic(graph_embedding)
        return values.float()

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
