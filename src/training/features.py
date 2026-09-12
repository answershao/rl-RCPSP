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


RESOURCE_CONTEXT_DIM = 16
RESOURCE_TIME_HIDDEN_DIM = 32


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
        # Standard GIN recipe around the residual stream.  Mean aggregation
        # alone only bounds the *degree* factor; the update's ReLU output still
        # feeds the next layer unnormalised, so depth and the residual sum
        # (1 + eps) * h + messages compound.  LayerNorm removes that drift,
        # which is what makes the same weights portable across suites as
        # different as j30 and RG300.
        self.norm = nn.LayerNorm(embedding_dim)

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
        return self.norm(embeddings + self.update(aggregate))


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
        max_horizon: int,
        static_cache: StaticGraphCache,
        max_successors: int = 20,
        gin_layers: int = 3,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        global_dim: int = 16,
        mixed_precision: str = "none",
    ) -> None:
        if gin_layers < 1:
            raise ValueError("gin_layers must be positive")
        self.layout = ObservationLayout(max_activities, max_resources, max_horizon)
        if observation_space.shape != (self.layout.size,):
            raise ValueError(
                f"expected flattened RCPSP observation {(self.layout.size,)}, "
                f"got {observation_space.shape}"
            )
        resource_context_dim = RESOURCE_CONTEXT_DIM
        features_dim = (
            max_activities * embedding_dim
            + global_dim
            + 2 * max_activities
            + max_activities * resource_context_dim
        )
        super().__init__(observation_space, features_dim)
        self.max_activities = max_activities
        self.max_resources = max_resources
        self.max_horizon = max_horizon
        self.max_successors = max_successors
        self.embedding_dim = embedding_dim
        self.global_dim = global_dim
        self.resource_context_dim = resource_context_dim
        self.mixed_precision = mixed_precision
        self.set_static_cache(static_cache)

        global_input_dim = self.layout.global_features.stop - self.layout.global_features.start
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, global_dim),
            nn.ReLU(),
        )
        activity_input_dim = 12 + DYNAMIC_ACTIVITY_FEATURE_COUNT + max_resources + global_dim
        self.activity_encoder = nn.Sequential(
            nn.Linear(activity_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.gin_layers = nn.ModuleList(
            DirectedGINLayer(embedding_dim, hidden_dim) for _ in range(gin_layers)
        )
        # Encode the resource calendar along time before matching it to each
        # candidate.  The temporal axis remains explicit; only the resource
        # channels are mixed by the convolution.
        self.resource_time_encoder = nn.Sequential(
            nn.Conv1d(
                max_resources,
                RESOURCE_TIME_HIDDEN_DIM,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                RESOURCE_TIME_HIDDEN_DIM,
                resource_context_dim,
                kernel_size=1,
            ),
            nn.ReLU(),
        )
        self.resource_time_key = nn.Linear(resource_context_dim + 1, resource_context_dim)
        self.resource_time_value = nn.Linear(resource_context_dim + 1, resource_context_dim)
        self.resource_query = nn.Linear(
            embedding_dim + max_resources, resource_context_dim
        )
        self.register_buffer(
            "resource_time_positions",
            th.arange(max_horizon, dtype=th.float32)
            / float(max(max_horizon - 1, 1)),
            persistent=False,
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
        for name, array in (
            ("slack_ratios", static_cache.slack_ratios),
            ("on_critical_path", static_cache.on_critical_path),
        ):
            if array.shape != expected_node_shape:
                raise ValueError(f"static cache {name} does not match max_activities")
        arrays = {
            "static_durations": static_cache.durations,
            "static_resource_demands": static_cache.resource_demands,
            "static_successor_indices": static_cache.successor_indices,
            "static_successor_counts": static_cache.successor_counts,
            "static_predecessor_counts": static_cache.predecessor_counts,
            "static_downstream_durations": static_cache.downstream_durations,
            "static_activity_mask": static_cache.activity_mask,
            "static_slack_ratios": static_cache.slack_ratios,
            "static_on_critical_path": static_cache.on_critical_path,
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

    def _candidate_resource_context(
        self,
        observations: th.Tensor,
        embeddings: th.Tensor,
        demands: th.Tensor,
        activity_mask: th.Tensor,
    ) -> th.Tensor:
        """Match each activity demand against the future resource calendar."""
        layout = self.layout
        batch_size = observations.shape[0]
        profile = observations[:, layout.resource_profile].reshape(
            batch_size, self.max_resources, self.max_horizon
        )
        temporal = self.resource_time_encoder(profile).transpose(1, 2)
        positions = self.resource_time_positions.to(dtype=temporal.dtype)
        positions = positions.view(1, self.max_horizon, 1).expand(batch_size, -1, -1)
        temporal = th.cat([temporal, positions], dim=-1)
        keys = self.resource_time_key(temporal)
        values = self.resource_time_value(temporal)

        query_input = th.cat([embeddings, demands], dim=-1)
        queries = self.resource_query(query_input)
        attention_logits = th.einsum("bnd,btd->bnt", queries, keys)
        attention_logits = attention_logits / float(self.resource_context_dim) ** 0.5

        # The flattened observation pads shorter instance calendars with zero;
        # prevent padded time slots from receiving attention mass.
        local_horizon = th.round(
            observations[:, layout.horizon] * float(self.max_horizon)
        ).long().clamp(min=1, max=self.max_horizon)
        time_indices = th.arange(self.max_horizon, device=observations.device)
        time_mask = time_indices.unsqueeze(0) < local_horizon.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(
            ~time_mask.unsqueeze(1), th.finfo(attention_logits.dtype).min
        )
        weights = th.softmax(attention_logits, dim=-1)
        context = th.einsum("bnt,btd->bnd", weights, values)
        return context * activity_mask.unsqueeze(-1)

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
        remaining_predecessors = observations[:, layout.remaining_predecessors]
        start_times = observations[:, layout.scheduled_start_times]
        finish_times = observations[:, layout.scheduled_finish_times]
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
        slack_ratios = self.static_slack_ratios[instance_indices]
        on_critical_path = self.static_on_critical_path[instance_indices]
        activity_mask = self.static_activity_mask[instance_indices]
        global_embedding = self.global_encoder(observations[:, layout.global_features])

        global_by_activity = global_embedding.unsqueeze(1).expand(-1, n, -1)
        activity_feature_parts = [
            status.unsqueeze(-1),
            precedence.unsqueeze(-1),
            remaining_predecessors.unsqueeze(-1),
            start_times.unsqueeze(-1),
            finish_times.unsqueeze(-1),
            durations.unsqueeze(-1),
            eligible.unsqueeze(-1),
            successor_counts.unsqueeze(-1),
            predecessor_counts.unsqueeze(-1),
            downstream_durations.unsqueeze(-1),
            slack_ratios.unsqueeze(-1),
            on_critical_path.unsqueeze(-1),
        ]
        activity_feature_parts.extend([demands, dynamic, global_by_activity])
        activity_features = th.cat(activity_feature_parts, dim=-1)
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

        resource_context = self._candidate_resource_context(
            observations, embeddings, demands, activity_mask
        )
        return th.cat(
            [
                embeddings.reshape(-1, n * self.embedding_dim),
                global_embedding,
                activity_mask,
                eligible,
                resource_context.reshape(-1, n * self.resource_context_dim),
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


class _CandidateSelfAttention(nn.Module):
    """One-layer masked self-attention with pre-LN and a residual connection.

    Inserts a small correction on top of the per-node actor input so that legal
    activities can exchange information before the per-node scoring MLP. The
    mask range is the legal set (activity_mask & eligible), keeping illegal
    activities out of softmax and out of the final logits (which are still
    masked to -inf downstream). Small init keeps the residual branch near
    zero at step 0, so training begins from a behaviour close to the baseline
    and PPO can learn how much to rely on it.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layer_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.output = nn.Linear(dim, dim, bias=True)
        # Small init keeps the branch close to identity at step 0.
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.output.bias)
        # P0v2: learnable gate. sigmoid(0)=0.5 at step 0, PPO can move toward
        # 0 (close attention branch) or 1 (adopt attention fully). Pure addition
        # in P0 made the residual branch a pure noise injection that PPO never
        # learned to amplify within 5M steps; gate lets PPO self-regulate.
        self.gate = nn.Parameter(th.tensor(0.0))

    def forward(self, x: th.Tensor, legal_mask: th.Tensor) -> th.Tensor:
        # x: (B, N, D); legal_mask: (B, N) bool, True means legal.
        batch_size, num_nodes, dim = x.shape
        y = self.layer_norm(x)
        qkv = self.qkv(y).reshape(
            batch_size, num_nodes, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(0)
        attn = queries @ keys.transpose(-2, -1)
        attn = attn / (self.head_dim ** 0.5)
        attn = attn.masked_fill(
            ~legal_mask.unsqueeze(1).unsqueeze(2),
            th.finfo(attn.dtype).min,
        )
        # Defensive guard for terminated envs where every key is masked out;
        # softmax would NaN otherwise. The corresponding logits are masked to
        # -inf below, so the uniform-distribution fallback does not affect
        # the categorical sample.
        all_illegal = ~legal_mask.any(dim=-1)  # (B,)
        if all_illegal.any():
            attn = attn.masked_fill(all_illegal.view(-1, 1, 1, 1), 0.0)
        attn = attn.softmax(dim=-1)
        attended = (attn @ values).transpose(1, 2).reshape(
            batch_size, num_nodes, dim
        )
        return x + th.sigmoid(self.gate) * self.output(attended)


class GINActorCriticHeads(nn.Module):
    """Independent node-scoring actor and graph-pooling critic."""

    def __init__(
        self,
        *,
        max_activities: int,
        embedding_dim: int,
        global_dim: int,
        resource_context_dim: int = RESOURCE_CONTEXT_DIM,
        hidden_dim: int = 64,
        mixed_precision: str = "none",
    ) -> None:
        super().__init__()
        self.max_activities = max_activities
        self.embedding_dim = embedding_dim
        self.global_dim = global_dim
        self.resource_context_dim = resource_context_dim
        self.mixed_precision = mixed_precision
        self.latent_dim_pi = max_activities
        self.latent_dim_vf = 1
        self.actor = nn.Sequential(
            nn.Linear(embedding_dim + resource_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # P0: candidate-set attention. candidate_dim must be divisible by
        # num_heads; 32 (emb) + 16 (resource ctx) = 48 = 4 * 12.
        candidate_dim = embedding_dim + resource_context_dim
        self.candidate_attention = _CandidateSelfAttention(candidate_dim, 4)
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
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        n = self.max_activities
        embedding_stop = n * self.embedding_dim
        global_stop = embedding_stop + self.global_dim
        mask_stop = global_stop + n
        embeddings = features[:, :embedding_stop].reshape(-1, n, self.embedding_dim)
        global_embedding = features[:, embedding_stop:global_stop]
        activity_mask = features[:, global_stop:mask_stop]
        eligible = features[:, mask_stop:mask_stop + n]
        context_start = mask_stop + n
        context_stop = context_start + n * self.resource_context_dim
        resource_context = features[:, context_start:context_stop].reshape(
            -1, n, self.resource_context_dim
        )
        return embeddings, global_embedding, activity_mask, eligible, resource_context

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        with self._autocast(features):
            embeddings, _, activity_mask, eligible, resource_context = self._unpack(features)
            actor_input = th.cat([embeddings, resource_context], dim=-1)
            legal = (activity_mask > 0.5) & (eligible > 0.5)
            # P0: candidate-set self-attention so legal activities exchange
            # information before the per-node scoring MLP. Illegal activities
            # are excluded from softmax and contribute no logits (masked to
            # -inf below). The residual branch is initialised near zero, so
            # the first training step behaves like the baseline.
            actor_input = self.candidate_attention(actor_input, legal)
            logits = self.actor(actor_input).squeeze(-1)
            logits = logits.masked_fill(~legal, th.finfo(logits.dtype).min)
        return logits.float()

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        with self._autocast(features):
            embeddings, global_embedding, activity_mask, _, _ = self._unpack(features)
            graph_embedding = pool_graph_embedding(
                embeddings, activity_mask, global_embedding
            )
            values = self.critic(graph_embedding)
        return values.float()

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
